import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_agent import Experiments
from experiment_control import set_paused
from archive.handoff_match_plan import boundary_ready, process_start
from archive.handoff_match_budget import boundary_ready as budget_boundary_ready
from match_plan import MATCH_NAMES, match_settings, read_match_plan
from archive.run_match_plan import PlannedQueue
from experiment_io import write
from archive.run_replay_followups import Queue


def plan(games=600, parallelism=300):
    return {'schema_version': 1, 'matches': {name: {'games': games, 'parallelism': parallelism}
                                            for name in MATCH_NAMES}}


class MatchPlanTests(unittest.TestCase):
    def test_mixed_budgets_require_explicit_equal_opponent_weighting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mixed = plan()
            mixed.update(schema_version=2, selection='mean-opponent-score-rate',
                         future_settings={'games': 1000, 'parallelism': 300})
            for name in MATCH_NAMES[2:]: mixed['matches'][name]['games'] = 1000
            write(root / 'match-plan.json', mixed)
            self.assertEqual(read_match_plan(root), mixed)
            snapshot = Experiments(root / 'activation', root).snapshot()
            self.assertEqual(snapshot['settings']['games'], 1000)
            phases = {p['id']: p for p in snapshot['phases']}
            self.assertEqual(phases['current-vs-deep64']['total'], 600)
            self.assertEqual(phases['wide64-vs-deep64']['total'], 1000)
            mixed.pop('selection')
            write(root / 'match-plan.json', mixed)
            with self.assertRaisesRegex(ValueError, 'equal opponent weighting'): read_match_plan(root)

    def test_resume_does_not_rerun_completed_training_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'match-plan.json', plan())
            queue = PlannedQueue(SimpleNamespace(output_dir=root, binary=Path('/tmp/alz')))
            run = root / 'trained'
            with patch.object(Queue, 'command') as command:
                queue.command('train', ['train-replay', '--run-dir', run])
                command.assert_called_once()
            write(run / 'result.json', {'completed_epochs': 20})
            with patch.object(Queue, 'command') as command:
                queue.command('train', ['train-replay', '--run-dir', run])
                command.assert_not_called()

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux process identities')
    def test_budget_handoff_waits_for_completed_match_and_idle_controller(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, pid = Path(temporary), os.getpid()
            settings = {'worker_pid': pid, 'worker_start': process_start(pid),
                        'active_stage': 'current-vs-deep64', 'next_stage': 'wide64-vs-deep64'}
            write(root / 'status.json', {'pid': pid, 'stage': settings['active_stage']})
            self.assertFalse(budget_boundary_ready(root, settings))
            set_paused(root, True)
            write(root / 'status.json', {'pid': pid, 'stage': 'paused', 'next_stage': settings['next_stage']})
            with self.assertRaisesRegex(RuntimeError, 'not published'): budget_boundary_ready(root, settings)
            write(root / 'current-vs-deep64.json', {})
            self.assertTrue(budget_boundary_ready(root, settings))
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            try:
                with self.assertRaisesRegex(RuntimeError, 'active child'): budget_boundary_ready(root, settings)
            finally:
                child.terminate(); child.wait(timeout=5)
            (root / 'wide64-vs-deep64.log').write_text('started')
            with self.assertRaisesRegex(RuntimeError, 'already started'): budget_boundary_ready(root, settings)

    def test_only_future_matches_change_and_equal_round_robin_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(read_match_plan(root))
            write(root / 'match-plan.json', plan())
            actual = read_match_plan(root)
            self.assertEqual(match_settings(actual, 'replay-relu-vs-selfplay', 300, 128),
                             {'games': 300, 'parallelism': 128})
            self.assertEqual(match_settings(actual, MATCH_NAMES[0], 300, 128),
                             {'games': 600, 'parallelism': 300})
            for invalid in [plan(601), plan(200, 300)]:
                write(root / 'match-plan.json', invalid)
                with self.assertRaises(ValueError): read_match_plan(root)
            invalid = plan()
            invalid['matches'][MATCH_NAMES[0]]['games'] = 300
            write(root / 'match-plan.json', invalid)
            with self.assertRaisesRegex(ValueError, 'equal budgets'): read_match_plan(root)

    def test_wrapper_changes_only_battle_settings_and_restores_training_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'match-plan.json', plan())
            args = SimpleNamespace(output_dir=root, binary=Path('/tmp/alz'), games=300, parallelism=128)
            queue = PlannedQueue(args)
            observed = []
            def battle(_self, name, *unused):
                observed.append((name, _self.args.games, _self.args.parallelism))
                return {}
            with patch.object(Queue, 'battle', battle):
                queue.battle('replay-relu-vs-selfplay', {}, {}, 1)
                queue.battle('current-vs-wide64', {}, {}, 2)
            self.assertEqual(observed, [('replay-relu-vs-selfplay', 300, 128), ('current-vs-wide64', 600, 300)])
            self.assertEqual((args.games, args.parallelism), (300, 128))
            with patch.object(Queue, 'battle', side_effect=RuntimeError('test failure')):
                with self.assertRaises(RuntimeError): queue.battle('current-vs-wide64', {}, {}, 2)
            self.assertEqual((args.games, args.parallelism), (300, 128))

    def test_dashboard_preserves_past_totals_and_shows_new_future_budgets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / 'queue'
            write(queue / 'match-plan.json', plan())
            write(queue / 'pooling-followup.json', {'directory': 'pooling'})
            report = {'config': {'games': 300, 'games_parallelism': 128, 'inference_batch_size': 64},
                      'games': [{}] * 300, 'first_temperature': 0.7,
                      **{key: {} for key in ['first_checkpoint', 'second_checkpoint',
                         'first_checkpoint_result', 'second_checkpoint_result']},
                      'duration_seconds': 5, 'average_plies': 10}
            write(queue / 'replay-relu-vs-selfplay.json', report)
            snapshot = Experiments(root / 'activation', queue).snapshot()
            phases = {p['id']: p for p in snapshot['phases']}
            self.assertEqual(phases['replay-relu-vs-selfplay']['total'], 300)
            self.assertEqual(phases['replay-relu-vs-selfplay']['match_settings']['parallelism'], 128)
            self.assertEqual(phases['current-vs-wide64']['total'], 600)
            self.assertEqual(phases['current-vs-katago-pooling']['total'], 600)
            self.assertEqual(snapshot['settings']['parallelism'], 300)

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux process identities')
    def test_handoff_requires_idle_boundary_and_rejects_started_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid = os.getpid()
            settings = {'worker_pid': pid, 'worker_start': process_start(pid), 'active_stage': 'train-wide'}
            write(root / 'status.json', {'pid': pid, 'stage': 'train-wide'})
            self.assertFalse(boundary_ready(root, settings))
            set_paused(root, True)
            write(root / 'status.json', {'pid': pid, 'stage': 'paused', 'next_stage': 'preflight-deep-inference'})
            self.assertTrue(boundary_ready(root, settings))
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            try:
                with self.assertRaisesRegex(RuntimeError, 'active child'): boundary_ready(root, settings)
            finally:
                process.terminate(); process.wait(timeout=5)
            write(root / 'current-vs-wide64.partial.json', {})
            with self.assertRaisesRegex(RuntimeError, 'already started'): boundary_ready(root, settings)


if __name__ == '__main__':
    unittest.main()
