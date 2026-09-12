import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recipe_battle_agent import RecipeExperiments
from archive.run_capacity_battles_overlap import can_start


class RecipeTests(unittest.TestCase):
    def test_overlap_waits_for_sixty_percent_and_limits_to_two(self):
        names = ['a', 'b', 'c', 'd']
        self.assertTrue(can_start(0, names, {}, {}, {}, 1000))
        self.assertFalse(can_start(1, names, {}, {'a': 1}, {'a': 599}, 1000))
        self.assertTrue(can_start(1, names, {}, {'a': 1}, {'a': 600}, 1000))
        self.assertFalse(can_start(2, names, {}, {'a': 1, 'b': 1}, {'b': 600}, 1000))
        self.assertTrue(can_start(2, names, {'b': 1}, {}, {}, 1000))

    def test_archive_history_and_live_progress_are_combined_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = {'snapshot': {'archived': True, 'phases': [{'id': 'old'}]},
                       'logs': {'old': {'lines': ['saved']}}}
            (root / 'archive.json').write_text(json.dumps(archive))
            plan = {'matches': [{'id': 'new', 'title': 'Control vs cosine'}],
                    'settings': {'games': 1000, 'parallelism': 300, 'inference_batch_size': 64}}
            (root / 'recipe-battle-plan.json').write_text(json.dumps(plan))
            (root / 'status.json').write_text(json.dumps({'active': ['new']}))
            (root / 'new.log').write_text('2026-09-08T21:00:00Z INFO games_completed=601 games_total=1000 elapsed_seconds=120\n')
            agent = RecipeExperiments(root, root)
            with patch('recipe_battle_agent.locked', return_value=True):
                snapshot = agent.snapshot()
            self.assertFalse(snapshot['archived'])
            self.assertEqual([p['id'] for p in snapshot['phases']], ['old', 'new'])
            phase = snapshot['phases'][-1]
            self.assertEqual((phase['state'], phase['completed'], phase['total']), ('running', 601, 1000))
            self.assertEqual(phase['match_settings']['parallelism'], 300)
            self.assertEqual(agent.dispatch({'method': 'logs', 'phase_id': 'old'}), archive['logs']['old'])
            for request in ({'method': 'pause'}, {'method': 'logs', 'phase_id': '../etc/passwd'}):
                with self.assertRaises(ValueError):
                    agent.dispatch(request)
            self.assertEqual(json.loads((root / 'archive.json').read_text()), archive)


if __name__ == '__main__':
    unittest.main()
