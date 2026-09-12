#!/usr/bin/env python3
"""Hand a paused queue to a new worker after its current comparison completes."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from experiment_control import read_control, set_paused
from experiment_io import checkpoint, digest, read, require, validate_battle, write



def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def boundary_ready(root, pid, expected_start):
    state = read(root / "status.json")
    require(process_start(pid) == expected_start, "Original queue worker exited or changed")
    require(state["pid"] == pid, "Queue worker identity changed")
    require(state["stage"] in ("replay-relu-vs-selfplay", "paused"),
            f"Unexpected stage during handoff: {state['stage']}")
    if state["stage"] != "paused":
        return False
    require(state.get("next_stage", "").startswith("preflight-"),
            "Worker is not paused before the first value-head preflight")
    require((root / "replay-relu-vs-selfplay.json").exists(), "Comparison has no complete report")
    children = Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
    require(not children, "Paused worker still has a running child; refusing to stop it")
    return True


def validate_backend(command, repo, log_path, timeout):
    """Bound validation time and reap its process group before allowing fallback."""
    with log_path.open("w") as log:
        try:
            process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=log,
                                       start_new_session=True)
        except OSError as error:
            return {"passed": False, "error": str(error), "timed_out": False}
        try:
            code = process.wait(timeout=timeout)
            return {"passed": code == 0, "returncode": code, "timed_out": False}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            # The shell can exit before a descendant; clear its remaining group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return {"passed": False, "returncode": process.returncode, "timed_out": True}


def resume_previous_trainer(root, plan):
    require(boundary_ready(root, plan["old_worker_pid"], plan["old_worker_start"]),
            "Previous worker is not at the safe fallback boundary")
    require(digest(root / "queue-config.json") == plan["previous_config_sha256"],
            "Cannot fall back after the queue configuration changes")
    config = read(root / "queue-config.json")
    require(digest(plan["previous_binary"]) == config["binary_sha256"],
            "Previous training executable changed")
    require(digest(plan["previous_worker_script"]) == config["script_sha256"],
            "Previous worker script changed")
    require(not any(root.glob("preflight-*")) and not any(root.glob("kata*")),
            "Cannot change backends after value-head work starts")
    set_paused(root, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--resume-gated-handoff", action="store_true",
                        help="Replace the waiting handoff while retaining its existing queue gate")
    args = parser.parse_args()
    plan = read(args.plan)
    root, repo = Path(plan["queue_dir"]), Path(plan["repository"])
    state_file = root / "training-handoff.json"

    def status(stage, **details):
        value = {"stage": stage, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "pid": os.getpid(), "plan": str(args.plan.resolve()), **details}
        write(state_file, value)
        print(value, flush=True)

    monitors = []
    try:
        with (root / ".handoff.lock").open("w") as handoff_lock:
            fcntl.flock(handoff_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(digest(root / "queue-config.json") == plan["previous_config_sha256"],
                    "Queue configuration changed before handoff")
            require(process_start(plan["old_worker_pid"]) == plan["old_worker_start"],
                    "Original queue worker identity changed")
            require(digest(repo / "target/release/alz") == plan["training_binary_sha256"],
                    "Prepared training binary changed")
            require(digest(repo / "scripts/archive/run_replay_followups.py") == plan["worker_script_sha256"],
                    "Prepared queue script changed")
            state = read(root / "status.json")
            if args.resume_gated_handoff:
                require(read_control(root)["paused"], "Expected the existing handoff gate")
                require(state["stage"] in ("replay-relu-vs-selfplay", "paused"),
                        "Worker is past the handoff boundary")
            else:
                require(state["stage"] == "replay-relu-vs-selfplay", "Comparison is no longer running")
                require(not read_control(root)["paused"], "Queue is already paused")
                set_paused(root, True)  # This gate is checked only between commands.
            status("waiting_for_current_match", current_match_uninterrupted=True)
            while not boundary_ready(root, plan["old_worker_pid"], plan["old_worker_start"]):
                time.sleep(5)
            config = read(root / "queue-config.json")
            first = checkpoint(read(Path(config["activation_dir"]) / "kata-v1/result.json")["checkpoint"]["path"])
            report = read(root / "replay-relu-vs-selfplay.json")
            validate_battle(report, first, config["latest_checkpoint"], config["games"], config["simulations"], 0.7)
            status("validating_cuda_backend")
            validation = validate_backend(plan["validation_command"], repo,
                                          root / "training-backend-validation.log",
                                          plan.get("validation_timeout_seconds", 300))
            write(root / "training-backend-validation.json", {
                **validation, "training_binary_sha256": plan["training_binary_sha256"],
                "command": plan["validation_command"],
                "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            if not validation["passed"] and plan.get("on_validation_failure") == "previous_trainer":
                resume_previous_trainer(root, plan)
                status("fallback_to_previous_trainer", validation=validation,
                       adam_backend="standard", replay_cache="none", old_worker_pid=plan["old_worker_pid"])
                return
            require(validation["passed"], "CUDA validation failed; training remains gated")
            require(boundary_ready(root, plan["old_worker_pid"], plan["old_worker_start"]),
                    "Worker left its safe handoff boundary")
            require(digest(repo / "target/release/alz") == plan["training_binary_sha256"],
                    "Training binary changed during validation")
            status("replacing_idle_worker")
            os.kill(plan["old_worker_pid"], signal.SIGTERM)
            # Its launcher must finish cleanup before starting replacement monitors.
            deadline = time.monotonic() + 20
            while process_start(plan["old_worker_pid"]) is not None or process_start(plan["old_launcher_pid"]) is not None:
                require(time.monotonic() < deadline, "Old worker or launcher did not exit")
                time.sleep(0.2)
            with (root / ".lock").open("a") as queue_lock:
                fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for name in ("exit-code.txt", "finished-at.txt", "launcher-started-at.txt", "gpu.csv", "host.csv"):
                    path = root / name
                    if path.exists():
                        backup = root / ("before-training-upgrade-" + name)
                        require(not backup.exists(), f"Existing handoff archive: {backup}")
                        path.rename(backup)
            for command in plan.get("monitor_commands", []):
                monitors.append(subprocess.Popen(command, cwd=repo))
            set_paused(root, False)
            status("optimized_worker_running", training_binary_sha256=plan["training_binary_sha256"])
            result = subprocess.run(plan["worker_command"], cwd=repo).returncode
            (root / "exit-code.txt").write_text(str(result) + "\n")
            (root / "finished-at.txt").write_text(datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
            require(result == 0, "Optimized queue worker failed; see status.json and its log")
            status("complete")
    except Exception as error:
        status("failed", error=str(error))
        raise
    finally:
        for monitor in monitors:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()


if __name__ == "__main__":
    main()
