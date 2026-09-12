import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from archive.run_board_mask_experiment import memory_reason, Experiment, CapacityPause


class BoardMaskGuardTests(unittest.TestCase):
    def test_host_and_gpu_limits_independently_pause(self):
        safe = {'working_bytes':80,'memory_limit':100,'gpu_memory_mib':20000}
        self.assertIsNone(memory_reason(safe))
        self.assertIn('Host', memory_reason({**safe,'working_bytes':89}))
        self.assertIn('GPU', memory_reason({**safe,'gpu_memory_mib':22001}))

    def test_guard_stops_owned_child_on_pressure(self):
        safe = {'working_bytes':80,'memory_limit':100,'gpu_memory_mib':20000}
        high = {**safe,'working_bytes':89}
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Experiment(Path(tmp),Path('/tmp/unused'))
            try:
                with patch('archive.run_board_mask_experiment.resource_sample',side_effect=[safe,high]), \
                     patch('archive.run_board_mask_experiment.subprocess.Popen') as spawn, \
                     patch('archive.run_board_mask_experiment.stop') as stop:
                    child=spawn.return_value
                    child.pid=1234
                    child.poll.return_value=None
                    with self.assertRaises(CapacityPause):
                        experiment.command('training',['unused'],'training.log')
                    stop.assert_called_once_with(child)
                    self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            finally:
                experiment.telemetry.close()


if __name__ == '__main__':
    unittest.main()
