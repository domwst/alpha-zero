import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from recipe_battle_agent import RecipeExperiments
from self_play_dashboard import append_self_play, epoch_game_metrics, self_play_duration, current_epoch_progress


class SelfPlayDashboardTests(unittest.TestCase):
    def test_live_epoch_transitions_and_completed_boundary(self):
        heartbeat = 'self-play progress games_completed=856 games_total=1000 completed_evaluations=99269448 elapsed_seconds=4020'
        current = current_epoch_progress([heartbeat], 29, 'running')
        self.assertEqual((current['epoch'],current['games_completed'],current['stage']), (29,856,'self_play'))
        trained = current_epoch_progress([heartbeat, 'self-play complete epoch=28 self_play_seconds=6100'], 29, 'running')
        self.assertEqual((trained['games_completed'],trained['stage'],trained['elapsed_seconds']), (1000,'training',6100))
        saving = current_epoch_progress([heartbeat,'training complete epoch=28'],29,'running')
        self.assertEqual(saving['stage'],'saving')
        self.assertIsNone(current_epoch_progress([heartbeat,'epoch complete epoch=28'],29,'running'))
        next_epoch = current_epoch_progress([heartbeat,'epoch complete epoch=28'],30,'running')
        self.assertIsNone(next_epoch['games_completed'])

    def test_live_epoch_ignores_old_heartbeats_paused_jobs_and_restored_attempts(self):
        import datetime
        after = datetime.datetime.fromisoformat('2026-09-11T09:00:00+00:00').timestamp()
        old = '2026-09-11T08:59:59Z self-play progress games_completed=1000 games_total=1000 completed_evaluations=100 elapsed_seconds=10'
        self.assertIsNone(current_epoch_progress([old],29,'running',after=after)['games_completed'])
        self.assertIsNone(current_epoch_progress([old],29,'paused'))
        self.assertIsNone(current_epoch_progress([old],29,'cancelled'))
        reset = current_epoch_progress([old,'restoring complete training snapshot snapshot_epoch=27'],29,'running')
        self.assertIsNone(reset['games_completed'])
        replay = current_epoch_progress(['starting replay-history epoch epoch=1','encoding replay cache'],2,'running',history_epochs=20)
        self.assertEqual(replay['stage'],'replay_training')
        self.assertIsNone(replay['games_completed'])

    def test_inherited_timing_requires_matching_provenance_and_keeps_native_timing(self):
        row = {'epoch':0, 'self_play_seconds':None, 'history_stats_sha256':'source-hash'}
        inherited = {'epochs':{'0':{'source_stats_sha256':'source-hash', 'self_play_seconds':123.5}}}
        self.assertEqual(self_play_duration(row, inherited), 123.5)
        self.assertEqual(self_play_duration({**row, 'self_play_seconds':50}, inherited), 50)
        self.assertIsNone(self_play_duration({**row, 'history_stats_sha256':'changed'}, inherited))
        self.assertIsNone(self_play_duration({**row, 'epoch':1}, inherited))
        self.assertIsNone(self_play_duration(row, {}))

    def test_win_rate_excludes_draws_from_wins_but_keeps_all_games_in_denominator(self):
        row = {'epoch':2,'games':10,'average_game_length':20.5,'total_game_length':205,'total_score':2}
        outcomes = {'epoch':2,'games':10,'first_player_wins':4,'second_player_wins':2,'draws':4,'total_game_length':205}
        metrics = epoch_game_metrics(row, outcomes)
        self.assertEqual(metrics['first_player_win_rate'], .4)
        self.assertEqual(metrics['draws'], 4)
        self.assertEqual(metrics['average_game_length'], 20.5)
        self.assertIsNone(epoch_game_metrics(row)['first_player_win_rate'])
        self.assertIsNone(epoch_game_metrics(row, {**outcomes,'epoch':1})['first_player_win_rate'])
        self.assertIsNone(epoch_game_metrics(row, {**outcomes,'first_player_wins':6})['first_player_win_rate'])
        self.assertIsNone(epoch_game_metrics(row, {**outcomes,'first_player_wins':2,'second_player_wins':4})['first_player_win_rate'])

    def test_progress_uses_complete_snapshots_and_has_no_fake_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent/'selfplay-nucleus'
            queue = parent/'recipe-battles'
            queue.mkdir()
            (root/'nucleus-p095/stats').mkdir(parents=True)
            (root/'jobs').mkdir()
            (root/'plan.json').write_text(json.dumps({'epochs':100,'inference_batch_size':256,
                'jobs':[{'id':'nucleus-p095','title':'Nucleus','top_p':0.95,'parallelism':400}]}))
            (root/'status.json').write_text('{}')
            (root/'jobs/nucleus-p095.json').write_text(json.dumps({'stage':'running','started_at':'2026-09-09T00:00:00+00:00'}))
            (root/'nucleus-p095/stats/00000000.json').write_text(json.dumps({'epoch':0,'epoch_seconds':100,
                'scheduled_learning_rate':0.0007493113525037345, 'config':{'learning_rate':0.001},
                'self_play_seconds':80.25, 'training':{'value_loss':0.5,'policy_loss':3.2,'duration_seconds':12.5,'samples':123456}}))
            (root/'nucleus-p095/stdout.log').write_text('self-play progress games_completed=12 games_total=1000 completed_evaluations=100000 elapsed_seconds=20\n')
            agent = RecipeExperiments(queue, queue)
            with patch('self_play_dashboard.locked', return_value=True):
                first = append_self_play(agent, {'phases':[]})
                self.assertEqual(first['phases'][0]['completed'], 0)
                self.assertEqual(first['self_play']['parallelisms'], [400])
                snapshot = root/'nucleus-p095/checkpoints/00000000'
                snapshot.mkdir(parents=True)
                (snapshot/'metadata.json').write_text('{}')
                actual = append_self_play(agent, {'phases':[]})['phases'][0]
            self.assertEqual(actual['completed'], 1)
            self.assertIsNone(actual['metrics'][0]['validation'])
            self.assertEqual(actual['metrics'][0]['self_play_seconds'], 80.25)
            self.assertEqual(actual['metrics'][0]['training']['duration_seconds'], 12.5)
            self.assertEqual(actual['metrics'][0]['training']['samples'], 123456)
            self.assertEqual(actual['metrics'][0]['scheduled_learning_rate'], 0.0007493113525037345)
            self.assertEqual(actual['current_epoch']['games_completed'], 12)
            self.assertEqual(actual['current_epoch']['evaluations_per_second'], 5000)
            self.assertEqual(actual['current_epoch']['epoch'], 2)
            self.assertIn('S3000/P400/B256', actual['note'])
            self.assertEqual(actual['state'], 'running')
            self.assertIn('nucleus-p095', agent.logs)
            overrides = root/'nucleus-p095/continuation-settings.json'
            overrides.write_text(json.dumps({'schema_version':1,'games_parallelism':300}))
            revised = append_self_play(agent, {'phases':[]})
            self.assertIn('S3000/P300/B256', revised['phases'][0]['note'])
            self.assertEqual(revised['self_play']['parallelisms'], [300])
            self.assertEqual(json.loads((root/'plan.json').read_text())['jobs'][0]['parallelism'], 400)



if __name__ == '__main__':
    unittest.main()
