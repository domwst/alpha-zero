from pathlib import Path
import subprocess
import sys
import unittest

from run_self_play_experiments import AdoptedProcess, process_identity, initial_job_state
from self_play_runtime import command, stop


class SupervisorTests(unittest.TestCase):
    def test_cancellation_survives_restart_and_takes_precedence_over_pause(self):
        old = {'stage':'paused', 'started_at':'2026-09-09T13:31:42+00:00'}
        cancelled_at = '2026-09-10T18:45:00+00:00'
        for paused in (True, False):
            state = initial_job_state({'cancelled':True, 'cancelled_at':cancelled_at,
                                       'paused':paused}, old)
            self.assertEqual(state['stage'], 'cancelled')
            self.assertEqual(state['started_at'], old['started_at'])
            self.assertEqual(state['ended_at'], cancelled_at)

    def test_user_hold_is_not_runnable_and_preserves_history(self):
        old = {'stage':'queued', 'started_at':'2026-09-09T13:31:42+00:00', 'pid':123}
        state = initial_job_state({'paused':True}, old)
        self.assertEqual(state['stage'], 'paused')
        self.assertEqual(state['started_at'], old['started_at'])
        self.assertIn('automatic resume disabled', state['note'])
        self.assertEqual(initial_job_state({'paused':False}, state), {'stage':'queued'})

    def worker(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                   start_new_session=True, stderr=subprocess.DEVNULL)
        self.addCleanup(self.cleanup, process)
        return process

    @staticmethod
    def cleanup(process):
        if process.poll() is None:
            process.kill()
        process.wait()

    def test_adoption_keeps_worker_and_stop_targets_its_group(self):
        process, other = self.worker(), self.worker()
        adopted = AdoptedProcess(process_identity(process.pid))
        self.assertIsNone(adopted.poll())
        with self.assertRaises(subprocess.TimeoutExpired):
            adopted.wait(timeout=.01)
        stop(adopted)
        self.assertEqual(adopted.wait(timeout=2), 0)
        self.assertIsNone(other.poll())

    def test_stale_pid_identity_is_rejected(self):
        process = self.worker()
        identity = process_identity(process.pid)
        identity['start_ticks'] = '0'
        with self.assertRaises(RuntimeError):
            AdoptedProcess(identity)
        self.assertIsNone(process.poll())

    def test_adopted_completion_detects_unreaped_exit(self):
        process = self.worker()
        adopted = AdoptedProcess(process_identity(process.pid))
        process.terminate()
        adopted.wait(timeout=2)
        self.assertIsNone(process_identity(process.pid))

    def test_control_parallelism_does_not_change_training_recipe(self):
        default = command(Path('/tmp/control'), 1.0, epochs=100, adam_backend='standard')
        resumed = command(Path('/tmp/control'), 1.0, epochs=100, adam_backend='standard', parallelism=400)
        index = default.index('--games-parallelism') + 1
        self.assertEqual(default[index], '500')
        self.assertEqual(resumed[index], '400')
        resumed[index] = '500'
        self.assertEqual(default, resumed)


if __name__ == '__main__':
    unittest.main()
