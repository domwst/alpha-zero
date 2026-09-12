import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiment_control import read_control, set_paused
from archive.handoff_replay_training import resume_previous_trainer, validate_backend
from experiment_io import digest, write



class FallbackTests(unittest.TestCase):
    def test_validation_success_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for code, passed in [('pass', True), ('raise SystemExit(3)', False)]:
                result = validate_backend([sys.executable, '-c', code], root, root/'test.log', 5)
                self.assertEqual(result['passed'], passed)
                self.assertFalse(result['timed_out'])
            # Include a descendant that ignores TERM: the group must be killed too.
            code = ('import subprocess,time; p=subprocess.Popen([' + repr(sys.executable) +
                    ',"-c","import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"]); '
                    'open("child.pid","w").write(str(p.pid)); time.sleep(60)')
            result = validate_backend([sys.executable, '-c', code], root, root/'test.log', 0.5)
            self.assertFalse(result['passed'])
            self.assertTrue(result['timed_out'])
            pid = int((root/'child.pid').read_text())
            # A killed descendant may briefly remain a zombie until init reaps it.
            import time
            for _ in range(50):
                path=Path(f'/proc/{pid}/stat')
                if not path.exists() or path.read_text().rsplit(')',1)[1].split()[0] == 'Z':
                    break
                time.sleep(0.02)
            else:
                self.fail('Timed-out validation left a live child')

    def test_fallback_resumes_original_worker_only_when_inputs_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            binary, script = root/'old-alz', root/'old-worker.py'
            binary.write_bytes(b'original binary'); script.write_bytes(b'original worker')
            config={'binary_sha256':digest(binary),'script_sha256':digest(script)}
            write(root/'queue-config.json',config)
            plan={'old_worker_pid':os.getpid(),'old_worker_start':'test',
                  'previous_binary':str(binary),'previous_worker_script':str(script),
                  'previous_config_sha256':digest(root/'queue-config.json')}
            set_paused(root, True)
            with patch('archive.handoff_replay_training.boundary_ready', return_value=True):
                binary.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'executable changed'):
                    resume_previous_trainer(root,plan)
                self.assertTrue(read_control(root)['paused'])
                binary.write_bytes(b'original binary')
                (root/'kata-gelu-value64-v1').mkdir()
                with self.assertRaisesRegex(RuntimeError, 'after value-head work starts'):
                    resume_previous_trainer(root,plan)
                (root/'kata-gelu-value64-v1').rmdir()
                resume_previous_trainer(root,plan)
                self.assertFalse(read_control(root)['paused'])
                self.assertEqual(json.loads((root/'queue-config.json').read_text()),config)


if __name__ == '__main__':
    unittest.main()
