from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from experiment_io import digest, read, write
from run_board_mask_selfplay import reconstruction_plan, continuation_command


class ReconstructionPlanTests(unittest.TestCase):
    def args(self, **updates):
        return Namespace(**{**dict(source_run=None, history_epochs=None, epochs=None,
                                  architecture=None, parallelism=None, wait_for_queue=None), **updates})

    def test_frozen_legacy_plan_resumes_without_rewriting_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'trainer'; binary.write_bytes(b'pinned trainer')
            saved = dict(binary_sha256=digest(binary), source='/original/source', history_epochs=20,
                         continue_to_epoch=100, command=['original', '--arguments'])
            write(root/'reconstruction-plan.json', saved)
            self.assertEqual(reconstruction_plan(root, binary, self.args()), saved)
            with self.assertRaisesRegex(RuntimeError, 'saved'):
                reconstruction_plan(root, binary, self.args(parallelism=400))
            binary.write_bytes(b'changed trainer')
            with self.assertRaisesRegex(RuntimeError, 'trainer changed'):
                reconstruction_plan(root, binary, self.args())

    def test_new_plan_inherits_source_recipe_and_records_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'trainer'; binary.write_bytes(b'trainer')
            source = root/'source'
            write(source/'stats/00000001.json', {'config': {
                'epochs':100, 'seed':123, 'learning_rate':.002, 'replay_lr_exponent':1.2,
                'replay_positions':62500, 'replay_games':None, 'training_batch_size':128,
                'games_parallelism':400, 'top_p':.95, 'bn_gamma_one':True}})
            plan = reconstruction_plan(root, binary, self.args(source_run=source, history_epochs=2,
                                       architecture='kata-gelu-v1', parallelism=500))
            argv = plan['command']
            for flag, value in [('seed','123'), ('learning-rate','0.002'), ('training-batch-size','128'),
                                ('games-parallelism','500'), ('architecture','kata-gelu-v1')]:
                self.assertEqual(argv[argv.index('--'+flag)+1], value)
            self.assertNotIn('--replay-games', argv)
            self.assertIn('--bn-gamma-one', argv)
            self.assertEqual(plan['continue_to_epoch'], 100)
            self.assertEqual(read(root/'reconstruction-plan.json'), plan)

    def test_legacy_game_count_does_not_conflict_with_position_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'trainer'; binary.write_bytes(b'trainer')
            write(root/'source/stats/00000000.json', {'config': {
                'replay_games':2500, 'replay_positions':37500, 'replay_position_growth':2250,
                'replay_growth_start_epoch':15, 'replay_lr_exponent':None}})
            argv = reconstruction_plan(root, binary, self.args(source_run=root/'source', history_epochs=1))['command']
            self.assertIn('--replay-games', argv)
            self.assertNotIn('--replay-positions', argv)
            self.assertNotIn('--replay-position-growth', argv)

    def test_continuation_override_preserves_history_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {'command':['trainer','--epochs','20','--games-parallelism','500','--inference-batch-size','256'],
                    'continue_to_epoch':100}
            original = list(plan['command'])
            argv = continuation_command(root, plan, 400)
            self.assertEqual(argv[argv.index('--games-parallelism')+1], '400')
            self.assertEqual(argv[argv.index('--epochs')+1], '100')
            self.assertEqual(argv[argv.index('--inference-batch-size')+1], '256')
            self.assertEqual(plan['command'], original)
            self.assertEqual(continuation_command(root, plan), argv)
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                continuation_command(root, plan, 300)

    def test_continuation_rejects_invalid_parallelism(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'positive'):
                continuation_command(Path(directory), {}, 0)
