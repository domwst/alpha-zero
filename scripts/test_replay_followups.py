"""Safety checks for the paid-GPU queue; run with python3 -m unittest discover -s scripts."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from archive.handoff_replay_training import boundary_ready, process_start
from experiment_io import digest, fixed, validate_battle, write
from archive.run_replay_followups import Queue, select_activation, upgrade_queue_config


def report():
    return {
        "first_checkpoint": {"path": "/tmp/relu", "model_sha256": "relu"},
        "second_checkpoint": {"path": "/tmp/gelu", "model_sha256": "gelu"},
        "config": {"games": 2, "simulations": 8},
        "first_temperature": 0.7, "second_temperature": 0.7,
        "games": [
            {"game": 1, "first_seat": "first_checkpoint", "winner": "first_checkpoint"},
            {"game": 2, "first_seat": "second_checkpoint", "winner": "second_checkpoint"},
        ],
        "first_checkpoint_result": {"wins": 1, "losses": 1, "draws": 0, "score": 1.0},
        "second_checkpoint_result": {"wins": 1, "losses": 1, "draws": 0, "score": 1.0},
    }


class QueueTests(unittest.TestCase):
    def test_only_training_uses_new_binary_and_performance_flags(self):
        args = SimpleNamespace(output_dir=Path('/tmp/queue'), binary=Path('/old/alz'),
            training_binary=Path('/new/alz'), adam_backend='fused', replay_cache='device', prefetch_batches=0)
        queue = Queue(args)
        for arguments in [['battle'], ['benchmark', 'inference']]:
            command = queue.command_line(arguments)
            self.assertEqual(command[1], '/old/alz')
            self.assertNotIn('--adam-backend', command)
        for arguments in [['train-replay'], ['benchmark', 'training']]:
            command = queue.command_line(arguments)
            self.assertEqual(command[1], '/new/alz')
            self.assertEqual(command[-6:], ['--adam-backend', 'fused', '--replay-cache', 'device', '--prefetch-batches', '0'])

    def test_upgrade_preserves_old_config_and_rejects_changed_match_or_started_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'queue-config.json'
            old = {'schema_version': 1, 'binary_sha256': 'old', 'script_sha256': 'old-script', 'games': 300}
            new = {**old, 'schema_version': 2, 'script_sha256': 'new-script',
                   'training_binary_sha256': 'new', 'training_performance': {'adam_backend': 'fused'}}
            write(path, old)
            before = digest(path)
            with self.assertRaisesRegex(RuntimeError, 'comparison must finish'):
                upgrade_queue_config(root, new, before)
            write(root / 'replay-relu-vs-selfplay.json', {})
            for changed in [{**new, 'games': 200}, {**new, 'binary_sha256': 'new'}]:
                with self.assertRaisesRegex(RuntimeError, 'match settings'):
                    upgrade_queue_config(root, changed, before)
            started = root / 'kata-gelu-value64-v1'
            started.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'training has started'):
                upgrade_queue_config(root, new, before)
            started.rmdir()
            with self.assertRaisesRegex(RuntimeError, 'configuration changed'):
                upgrade_queue_config(root, new, 'wrong-digest')
            upgrade_queue_config(root, new, before)
            self.assertEqual(json.loads(path.read_text()), new)
            audit = json.loads((root / 'training-backend-upgrade.json').read_text())
            self.assertEqual(audit['previous'], old)
            upgrade_queue_config(root, new, before)  # Idempotent after an audited upgrade.

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux process identity check')
    def test_handoff_waits_for_completed_comparison_and_idle_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid = os.getpid()
            start = process_start(pid)
            state = {'pid': pid, 'stage': 'replay-relu-vs-selfplay'}
            write(root / 'status.json', state)
            self.assertFalse(boundary_ready(root, pid, start))
            with self.assertRaisesRegex(RuntimeError, 'exited or changed'):
                boundary_ready(root, pid, 'wrong-start-time')
            state.update(stage='paused', next_stage='preflight-kata-gelu-value64-v1-inference')
            write(root / 'status.json', state)
            with self.assertRaisesRegex(RuntimeError, 'complete report'):
                boundary_ready(root, pid, start)
            write(root / 'replay-relu-vs-selfplay.json', {})
            with patch('archive.handoff_replay_training.Path.read_text', autospec=True) as read_text:
                original = json.dumps(state)
                read_text.side_effect = lambda p: original if p.name == 'status.json' else '99999'
                with patch('archive.handoff_replay_training.process_start', return_value=start):
                    with self.assertRaisesRegex(RuntimeError, 'running child'):
                        boundary_ready(root, pid, start)

    def test_selection_counts_draws_and_breaks_ties_with_relu(self):
        data = report()
        self.assertEqual(select_activation(data), "kata-v1")
        data["second_checkpoint_result"]["score"] = 1.5
        self.assertEqual(select_activation(data), "kata-gelu-v1")
        data["first_checkpoint_result"]["score"] = 2.0
        self.assertEqual(select_activation(data), "kata-v1")

    def test_rejects_incomplete_wrong_or_inconsistent_matches(self):
        original = report()
        def validate(data):
            validate_battle(data, original["first_checkpoint"], original["second_checkpoint"], 2, 8, 0.7)
        validate(original)
        for change in [
            lambda r: r["games"].pop(),
            lambda r: r["games"][1].update(game=1),
            lambda r: r["games"][1].update(first_seat="first_checkpoint"),
            lambda r: r["config"].update(simulations=4),
            lambda r: r.update(first_temperature=0.0),
            lambda r: r["first_checkpoint"].update(model_sha256="other"),
            lambda r: r["first_checkpoint_result"].update(wins=2),
        ]:
            data = copy.deepcopy(original)
            change(data)
            with self.assertRaises(RuntimeError):
                validate(data)

    def test_config_changes_cannot_silently_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            fixed(path, {"binary": "a", "games": 300})
            fixed(path, {"binary": "a", "games": 300})
            with self.assertRaises(RuntimeError):
                fixed(path, {"binary": "b", "games": 300})

    def test_predecessor_failure_stops_before_any_gpu_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"test")
            model = root / "model.safetensors"
            model.write_bytes(b"test weights")
            write(root / "metadata.json", {
                "model_sha256": digest(model), "format_version": 2, "epoch": 69,
                "model": {"architecture": "kata_v1"}, "tensor_schema_sha256": "test",
            })
            success = root / "exit-code.txt"
            success.write_text("1\n")
            args = SimpleNamespace(output_dir=root, binary=binary, latest_checkpoint=root,
                activation_dir=root, device="cuda", games=300, simulations=4000,
                parallelism=128, inference_batch_size=64, require_success_file=success)
            queue = Queue(args)
            with patch.object(queue, "command") as command:
                with self.assertRaisesRegex(RuntimeError, "Activation experiment failed"):
                    queue.run()
                command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
