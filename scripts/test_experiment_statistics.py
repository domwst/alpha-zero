import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment_agent import Experiments
from experiment_statistics import completion_timing, distribution, summarize_match


class StatisticsTests(unittest.TestCase):
    def test_seats_are_relative_to_each_checkpoint_and_draws_count_once(self):
        a, b = 'first_checkpoint', 'second_checkpoint'
        games = [{'first_seat': first, 'second_seat': b if first == a else a, 'winner': winner, 'plies': length}
                 for first, winner, length in [(a, a, 9), (b, a, 20), (a, b, 31),
                                              (b, a, 40), (a, None, 361), (b, None, 200)]]
        with tempfile.TemporaryDirectory() as temporary:
            summary = summarize_match({'games': games, 'duration_seconds': 600}, Path(temporary) / 'missing.log')
        self.assertEqual(summary['seat_results'][a]['first'], {'games': 3, 'wins': 1, 'losses': 1, 'draws': 1})
        self.assertEqual(summary['seat_results'][a]['second'], {'games': 3, 'wins': 2, 'losses': 0, 'draws': 1})
        self.assertEqual(summary['seat_results'][b]['first'], {'games': 3, 'wins': 0, 'losses': 2, 'draws': 1})
        self.assertEqual(summary['outcomes'], {'first_player_wins': 1, 'second_player_wins': 3, 'draws': 2})
        self.assertEqual(summary['lengths'][a]['count'], 3)
        self.assertEqual(summary['lengths'][a]['median'], 20)
        self.assertEqual(summary['lengths']['draws']['mean'], 280.5)
        self.assertEqual(summary['lengths']['all']['median'], 35.5)
        self.assertEqual(summary['lengths']['all']['p90'], 361)
        self.assertEqual(sum(b['count'] for b in summary['lengths']['all']['histogram']), 6)
        self.assertEqual(summary['lengths']['all']['frequencies'], [
            {'moves': n, 'count': 1} for n in [9, 20, 31, 40, 200, 361]])
        self.assertIsNone(summary['completion'])
        self.assertFalse(summary['individual_game_durations_available'])

    def test_distribution_has_explicit_empty_and_single_game_behavior(self):
        self.assertIsNone(distribution([]))
        summary = distribution([20])
        for key in ('minimum', 'maximum', 'mean', 'median', 'p90', 'p95', 'p99'):
            self.assertEqual(summary[key], 20)
        self.assertEqual(summary['histogram'], [{'lower': 0, 'upper': 19, 'count': 0},
                                               {'lower': 20, 'upper': 39, 'count': 1}])
        self.assertIsNone(summarize_match({'games': [{}]}, Path('/tmp/missing')))
        self.assertEqual(distribution([1, 20, 1, 21])['frequencies'], [
            {'moves': 1, 'count': 2}, {'moves': 20, 'count': 1}, {'moves': 21, 'count': 1}])

    def test_completion_milestones_measure_series_time_and_last_ten_percent(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'battle.log'
            log.write_text(''.join(f'games_completed={i} games_total=10 elapsed_seconds={seconds}\n'
                                  for i, seconds in enumerate([0, 100, 120, 130, 140, 150, 170, 200, 220, 300, 600])))
            result = completion_timing(log, 10, 600.1)
            self.assertEqual(result['p50_seconds'], 150)
            self.assertEqual(result['p90_seconds'], 300)
            self.assertEqual(result['p99_seconds'], 600)
            self.assertEqual(result['final_10_percent_seconds'], 300)
            self.assertEqual(result['first_finish_seconds'], 100)
            self.assertIsNone(completion_timing(log, 10, 800))
            self.assertIsNone(completion_timing(log, 20, 600))

    def test_restart_and_incomplete_logs_cannot_mix_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'battle.log'
            old = 'games_completed=1 games_total=2 elapsed_seconds=10\ngames_completed=2 games_total=2 elapsed_seconds=20\n'
            log.write_text(old + 'games_completed=0 games_total=2 elapsed_seconds=1\n')
            self.assertIsNone(completion_timing(log, 2, 20))
            with log.open('a') as stream:
                stream.write('games_completed=1 games_total=2 elapsed_seconds=40\ngames_completed=2 games_total=2 elapsed_seconds=80\n')
            result = completion_timing(log, 2, 80)
            self.assertEqual(result['p50_seconds'], 40)
            self.assertEqual(result['last_finish_seconds'], 80)

    def test_large_curves_are_bounded_and_keep_milestones(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'battle.log'
            log.write_text(''.join(f'games_completed={i} games_total=1000 elapsed_seconds={i * 2}\n' for i in range(1001)))
            result = completion_timing(log, 1000, 2000)
            self.assertLessEqual(len(result['points']), 306)
            self.assertEqual(result['points'][-1], {'games': 1000, 'seconds': 2000})
            for key in ('p50_seconds', 'p90_seconds', 'p95_seconds', 'p99_seconds'):
                self.assertIn(result[key], [p['seconds'] for p in result['points']])

    def test_agent_caches_completed_statistics_and_invalidates_changed_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, log = root / 'battle.json', root / 'battle.log'
            output.write_text(json.dumps({'games': [{}], 'config': {}, 'duration_seconds': 10, 'average_plies': 5,
                                         **{k: {} for k in ('first_checkpoint', 'second_checkpoint', 'first_checkpoint_result', 'second_checkpoint_result')}}))
            log.write_text('initial')
            agent = Experiments(root, root)
            with patch('experiment_agent.summarize_match', return_value={'games': 1}) as summarize:
                for _ in range(2):
                    phase = agent.battle('match', 'Match', output, log, 'pending', 'models')
                    self.assertEqual(phase['result']['game_statistics'], {'games': 1})
                self.assertEqual(summarize.call_count, 1)
                log.write_text('changed log')
                agent.battle('match', 'Match', output, log, 'pending', 'models')
                self.assertEqual(summarize.call_count, 2)


if __name__ == '__main__':
    unittest.main()
