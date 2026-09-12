#!/usr/bin/env python3
"""Read-only JSON RPC agent over SSH stdin/stdout; no listening port."""

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import re
import select
import sys
import termios
import tty

from experiment_control import read_control
from experiment_io import now, locked, tail
from match_plan import match_settings, read_match_plan
from experiment_statistics import summarize_match, summarize_live_match
from experiment_metadata import apply_experiment_metadata, with_experiment_metadata




def log_timestamp(line):
    # Native logs use UTC RFC 3339 timestamps; tolerate ANSI-colored output.
    fields = re.sub(r'\x1b\[[0-9;]*m', '', line).split(maxsplit=1)
    try:
        value = datetime.datetime.fromisoformat(fields[0].replace('Z', '+00:00'))
        return value if value.tzinfo is not None else None
    except (ValueError, IndexError):
        return None


def job_timing(log, result_path, completed, running=False, reported_duration=None):
    started = None
    try:
        with Path(log).open('rb') as stream:
            lines = stream.read(65536).decode(errors='replace').splitlines()
        started = next((stamp for line in lines if (stamp := log_timestamp(line)) is not None), None)
    except FileNotFoundError:
        pass
    ended = None
    if completed:
        try:
            ended = datetime.datetime.fromtimestamp(Path(result_path).stat().st_mtime, datetime.timezone.utc)
        except FileNotFoundError:
            pass
    source = 'log_and_result' if started else None
    if started is None and ended is not None and isinstance(reported_duration, (int, float)) and 0 <= reported_duration < float('inf'):
        started = ended - datetime.timedelta(seconds=reported_duration)
        source = 'report_duration_and_result'
    boundary = ended or (datetime.datetime.now(datetime.timezone.utc) if running else None)
    duration = (boundary - started).total_seconds() if started and boundary else None
    return {'started_at': started.isoformat() if started else None,
            'ended_at': ended.isoformat() if ended else None,
            'duration_seconds': duration if duration is not None and duration >= 0 else None,
            'timing_source': source}


class Experiments:
    def __init__(self, activation_dir, queue_dir, epochs=20, games=300):
        self.activation = Path(activation_dir)
        self.queue = Path(queue_dir)
        self.epochs = epochs
        self.games = games
        self.cache = {}
        self.logs = {}
        self.match_statistics_cache = {}

    def json(self, path, default=None):
        path = Path(path)
        try:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
            cached = self.cache.get(path)
            if cached and cached[0] == key:
                return cached[1]
            value = json.loads(path.read_text())
            self.cache[path] = key, value
            return value
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def training(self, identity, title, directory, log, state, architecture):
        metrics = []
        for path in sorted(directory.glob("epochs/*.json")):
            data = self.json(path)
            if data is None or not (directory / "checkpoints" / path.stem / "metadata.json").exists():
                continue
            metrics.append({"epoch": data["epoch"] + 1, "training": data["training"],
                            "validation": data["validation"]})
        completed = len(metrics)
        if self.json(directory / "result.json"):
            state = "completed"
        lines = tail(log)
        starts = [line for line in lines if "starting fixed replay training pass" in line]
        pass_started = log_timestamp(starts[-1]) if starts else None
        duration = None
        if completed:
            times = [(directory / "checkpoints" / f'{m["epoch"] - 1:08}' / "metadata.json").stat().st_mtime
                     for m in metrics[-6:]]
            duration = ((times[-1] - times[0]) / (len(times) - 1) if len(times) > 1
                        else metrics[-1]["training"]["duration_seconds"])
        eta = None
        if state == "running" and duration:
            elapsed = 0
            if pass_started:
                elapsed = max(0, datetime.datetime.now(datetime.timezone.utc).timestamp() -
                              pass_started.timestamp())
            eta = max(0, (self.epochs - completed) * duration - elapsed)
        self.logs[identity] = log
        config = self.json(directory / "replay-config.json")
        cache = self.json(directory / "batch-cache.json", {})
        performance = ({"adam_backend": config.get("adam_backend", "standard"),
                        "replay_cache": cache.get("mode", "none"),
                        "prefetch_batches": cache.get("prefetch_batches", 0)} if config else None)
        return {"id": identity, "title": title, "kind": "training", "state": state,
                "architecture": architecture, "completed": completed, "total": self.epochs,
                "unit": "passes", "metrics": metrics, "eta_seconds": eta,
                "pass_seconds": duration,
                "current_pass_started_at": pass_started.isoformat() if pass_started else None,
                **job_timing(log, directory / "result.json", state == "completed", state == "running"),
                "result": None,
                "training_performance": performance}

    def battle(self, identity, title, output, log, state, architectures, settings=None):
        report = self.json(output)
        protocol = {"games": self.games, "parallelism": 128, "inference_batch_size": 64,
                    "simulations": 4000, "temperature": 0.7, **(settings or {})}
        progress = []
        for line in tail(log):
            match = re.search(r'games_completed=(\d+).*games_total=(\d+).*elapsed_seconds=([\d.]+)', line)
            if match:
                progress.append((int(match[1]), float(match[3])))
                protocol['games'] = int(match[2])
        completed, elapsed = progress[-1] if progress else (0, 0)
        result = None
        live_statistics = None
        if report:
            state, completed = "completed", len(report["games"])
            protocol.update({key: report['config'][key] for key in
                             ('games', 'simulations', 'inference_batch_size') if key in report['config']})
            protocol['parallelism'] = report['config'].get('games_parallelism', protocol['parallelism'])
            protocol['temperature'] = report.get('first_temperature', protocol['temperature'])
            result = {key: report[key] for key in ("first_checkpoint_result", "second_checkpoint_result",
                      "duration_seconds", "average_plies", "first_checkpoint", "second_checkpoint")}
            # Completed logs can be megabytes long. Recompute only when inputs change.
            key = tuple((path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None
                        for path in (output, log))
            cached = self.match_statistics_cache.get(identity)
            if cached is None or cached[0] != key:
                cached = key, summarize_match(report, log)
                self.match_statistics_cache[identity] = cached
            result['game_statistics'] = cached[1]
        elif state in ('running', 'paused', 'failed', 'unknown') and log.exists():
            stat = log.stat()
            key = ('live', str(log), stat.st_mtime_ns, stat.st_size, protocol['games'])
            cached = self.match_statistics_cache.get(identity)
            if cached is None or cached[0] != key:
                cached = key, summarize_live_match(log, protocol['games'])
                self.match_statistics_cache[identity] = cached
            live_statistics = cached[1]
        self.logs[identity] = log
        phase = {"id": identity, "title": title, "kind": "battle", "state": state,
                "architecture": architectures, "completed": completed, "total": protocol['games'],
                "match_settings": protocol,
                "unit": "games", "metrics": [], "result": result,
                "live_statistics": live_statistics,
                **job_timing(log, output, report is not None, state == "running",
                             report.get("duration_seconds") if report else None),
                "eta_seconds": (elapsed / completed * (protocol['games'] - completed)
                                if state == "running" and completed else None)}
        metadata = self.json(output.with_suffix('.metadata.json'), (report or {}).get('experiment_metadata', {}))
        return with_experiment_metadata(phase, metadata)

    def snapshot(self):
        activation_exit = self.activation / "exit-code.txt"
        activation_alive = locked(self.activation / ".lock")
        activation_failed = activation_exit.exists() and activation_exit.read_text().strip() != "0"
        stage = self.json(self.queue / "status.json", {})
        control = read_control(self.queue)
        queue_alive = locked(self.queue / ".lock")
        selection = self.json(self.queue / "selection.json")
        plan = read_match_plan(self.queue)
        future = match_settings(plan, 'current-vs-wide64', self.games, 128)
        if plan and 'future_settings' in plan:
            future = dict(plan['future_settings'])
        selected = selection["activation"] if selection else None
        prefix = "kata-gelu" if selected == "kata-gelu-v1" else "kata"
        variants = [prefix + "-value64-v1", prefix + "-value64x2-v1"]
        dataset = self.json(self.activation / "kata-v1/dataset.json", {})
        phases = []
        predecessor_complete = True
        for name, title in [("kata-v1", "Train fresh ReLU"), ("kata-gelu-v1", "Train fresh GELU")]:
            completed = self.json(self.activation / name / "result.json") is not None
            state = "pending"
            if predecessor_complete and not completed:
                state = "running" if activation_alive else "failed" if activation_failed else "unknown"
            phase = self.training(name, title, self.activation / name,
                                  self.activation / (name + ".log"), state, name)
            phases.append(phase)
            predecessor_complete = predecessor_complete and completed
        phases.append(self.battle("activation-match", "ReLU vs GELU", self.activation / "battle.json",
            self.activation / "battle.log", "running" if predecessor_complete and activation_alive else
            "failed" if predecessor_complete and activation_failed else "pending", "ReLU / GELU"))

        def state_for(command, stage=stage, alive=queue_alive):
            current = stage.get("stage", "")
            if current == command or current.startswith("preflight-" + command.removeprefix("train-")):
                return "running" if alive else "unknown"
            if current == "failed" and stage.get("failed_stage") == command:
                return "failed"
            if current == "paused" and stage.get("next_stage") == command:
                return "paused"
            return "pending"

        phases.append(self.battle("replay-relu-vs-selfplay", "Replay ReLU vs self-play checkpoint 69",
            self.queue / "replay-relu-vs-selfplay.json", self.queue / "replay-relu-vs-selfplay.log",
            state_for("replay-relu-vs-selfplay"), "ReLU / checkpoint 69"))
        for name, title in zip(variants, ["Train wide value head · 64 → 64 → 1",
                                         "Train deep value head · 64 → 64 → 64 → 1"]):
            phases.append(self.training("train-" + name, title, self.queue / name,
                self.queue / ("train-" + name + ".log"), state_for("train-" + name),
                name if selected else "Activation selected after ReLU/GELU match"))
        for name, title in [("current-vs-wide64", "Current vs wide value head"),
                            ("current-vs-deep64", "Current vs deep value head"),
                            ("wide64-vs-deep64", "Wide vs deep value head")]:
            phases.append(self.battle(name, title, self.queue / (name + ".json"),
                self.queue / (name + ".log"), state_for(name), selected or "Activation pending",
                match_settings(plan, name, self.games, 128)))
        pooling = self.queue / "pooling"
        pooling_selection = None
        if self.json(self.queue / "pooling-followup.json"):
            pooling_stage = self.json(pooling / "status.json", {})
            pooling_alive = locked(pooling / ".lock")
            pooling_selection = self.json(pooling / "selection.json")
            reuse = self.json(pooling / "queue-config.json", {}).get("baseline_policy") == "reuse-selected-checkpoint"
            architectures = (pooling_selection or {}).get("architectures", {})
            for name, title in [("current-pooling", "Train best value head · current pooling"),
                                ("katago-pooling", "Train best value head · KataGo pooling")]:
                if reuse and name == "current-pooling":
                    continue
                state = state_for("train-" + name, pooling_stage, pooling_alive)
                if name == ("katago-pooling" if reuse else "current-pooling") and pooling_stage.get("stage", "").startswith("validating_pooling"):
                    state = "running" if pooling_alive else "unknown"
                phases.append(self.training("train-" + name, title, pooling / name,
                    pooling / ("train-" + name + ".log"), state,
                    architectures.get(name, "Value head selected after round robin")))
                if reuse:
                    phases[-1]["note"] = "Reuses the winning value-head checkpoint for current pooling. Only the KataGo pooling model is trained here."
            name = "current-vs-katago-pooling"
            phases.append(self.battle(name, "Best value head · current vs KataGo pooling",
                pooling / (name + ".json"), pooling / (name + ".log"),
                state_for(name, pooling_stage, pooling_alive), "Current / KataGo pooling",
                match_settings(plan, name, self.games, 128)))
            if reuse:
                phases[-1]["note"] = "Current pooling uses the existing checkpoint selected by the value-head round robin."
                phases[-1]["baseline_checkpoint"] = (self.json(pooling / "reused-baseline.json", {}).get("checkpoint"))
            if stage.get("stage") == "complete":
                stage, queue_alive, control = pooling_stage, pooling_alive, read_control(pooling)
        comparisons = self.queue / "pooling-checkpoints"
        comparison_plan = self.json(comparisons / "plan.json")
        if comparison_plan:
            comparison_stage = self.json(comparisons / "status.json", {})
            comparison_alive = locked(comparisons / ".lock")
            cancellation = self.json(comparisons / "cancellation.json", {})
            for match in comparison_plan["matches"]:
                name = match["id"]
                phase = self.battle(name, match["title"], comparisons / (name + ".json"),
                    comparisons / (name + ".log"), state_for(name, comparison_stage, comparison_alive),
                    "GELU · deep value head · selected checkpoints", comparison_plan["settings"])
                phase["note"] = match["note"]
                if name in cancellation.get("jobs", []) and phase["result"] is None:
                    phase["state"] = "cancelled"
                    phase["eta_seconds"] = None
                    if phase.get("started_at"):
                        phase["ended_at"] = cancellation["cancelled_at"]
                        phase["duration_seconds"] = (datetime.datetime.fromisoformat(cancellation["cancelled_at"]) - datetime.datetime.fromisoformat(phase["started_at"])).total_seconds()
                    phase["note"] = "Cancelled by request. " + cancellation["reason"]
                    partial = cancellation.get("partial", {})
                    if partial.get("id") == name:
                        counts = partial["seat_results"]
                        wins = [sum(seat["wins"] for seat in counts[side].values()) for side in ("first_checkpoint", "second_checkpoint")]
                        phase["completed"] = partial["games_completed"]
                        phase["note"] += f' Completed games only: pass 18 {wins[0]} wins, pass 20 {wins[1]} wins. Unfinished long games are excluded; this is not a completed strength comparison.'
                phases.append(phase)
            if stage.get("stage") == "complete":
                stage, queue_alive, control = comparison_stage, comparison_alive, read_control(comparisons)
        capacity = self.queue / "capacity"
        capacity_plan = self.json(capacity / "schedule.json")
        if capacity_plan:
            capacity_stage = self.json(capacity / "status.json", {})
            capacity_alive = locked(capacity / ".lock")
            parallel = capacity / "parallel-width48"
            parallel_stage = self.json(parallel / "status.json", {})
            parallel_alive = locked(parallel / ".lock")
            overlap = capacity / "overlap-battles"
            overlap_plan = self.json(overlap / "plan.json")
            overlap_stage = self.json(overlap / "status.json", {})
            overlap_alive = locked(overlap / ".lock")
            for model in capacity_plan["models"]:
                name = "train-" + model["id"]
                state = state_for(name, capacity_stage, capacity_alive)
                if model == capacity_plan["models"][0] and capacity_stage.get("stage", "").startswith("validating_"):
                    state = "running" if capacity_alive else "unknown"
                if model["id"] == "capacity-width48" and parallel_stage and parallel_stage.get("stage") != "complete":
                    state = ("failed" if parallel_stage.get("stage") == "failed" else
                             "running" if parallel_alive else "unknown")
                phase = self.training(name, model["title"], capacity / model["id"],
                    capacity / (name + ".log"), state, model["architecture"])
                phase["note"] = capacity_plan["note"]
                if model["id"] == "capacity-width48" and parallel_stage:
                    phase["note"] += " Runs alongside depth training; comparisons wait for both models."
                    if parallel_stage.get("error"):
                        phase["note"] += " " + parallel_stage["error"]
                phases.append(phase)
            for match in capacity_plan["matches"]:
                name = match["id"]
                battle_state = state_for(name, capacity_stage, capacity_alive)
                job = self.json(overlap / "jobs" / (name + ".json"), {})
                if job and job.get("stage") != "complete":
                    battle_state = ("failed" if job.get("stage") == "failed" else
                                    "paused" if job.get("stage") == "paused" else
                                    "running" if overlap_alive else "unknown")
                phase = self.battle(name, match["title"], capacity / (name + ".json"),
                    capacity / (name + ".log"), battle_state,
                    "GELU · original pooling · deep value head", capacity_plan["settings"])
                phase["note"] = capacity_plan["note"]
                if overlap_plan:
                    policy = overlap_plan["policy"]
                    phase["note"] += f' Next comparison starts at {policy["start_next_at_percent"]}% completed games; at most {policy["max_active_battles"]} comparisons run together.'
                    if overlap_stage.get("error"):
                        phase["note"] += " Scheduler error: " + overlap_stage["error"]
                phases.append(phase)
            if stage.get("stage") in ("complete", "cancelled"):
                stage, queue_alive, control = capacity_stage, capacity_alive, read_control(capacity)
        third = self.queue / "third-global"
        third_plan = self.json(third / "schedule.json")
        if third_plan:
            third_stage = self.json(third / "status.json", {})
            third_alive = locked(third / ".lock")
            for model in third_plan["models"]:
                name = "train-" + model["id"]
                state = state_for(name, third_stage, third_alive)
                if third_stage.get("stage", "").startswith(("validating_", "preflight-", "building")):
                    state = "running" if third_alive else "unknown"
                phase = self.training(name, model["title"], third / model["id"],
                                      third / (name + ".log"), state, model["architecture"])
                phase["note"] = third_plan["note"]
                phases.append(phase)
            for match in third_plan["matches"]:
                name = match["id"]
                state = state_for(name, third_stage, third_alive)
                job = self.json(third / "overlap-battles/jobs" / (name + ".json"), {})
                if job and job.get("stage") != "complete":
                    state = ("failed" if job.get("stage") == "failed" else
                             "paused" if job.get("stage") == "paused" else
                             "running" if third_alive else "unknown")
                phase = self.battle(name, match["title"], third / (name + ".json"),
                                    third / (name + ".log"), state,
                                    "GELU · original pooling · deep value head", third_plan["settings"])
                phase["note"] = third_plan["note"] + " Second comparison starts at 60% completed games; at most two comparisons run together."
                phases.append(phase)
            stage, queue_alive, control = third_stage, third_alive, read_control(third)
        recipes = self.queue / "training-recipes"
        recipe_plan = self.json(recipes / "schedule.json")
        if recipe_plan:
            recipe_stage = self.json(recipes / "status.json", {})
            recipe_alive = locked(recipes / ".lock")
            for model in recipe_plan["models"]:
                name = model["id"]
                state = "queued"
                if name in recipe_stage.get("active", []):
                    state = "running" if recipe_alive else "unknown"
                elif recipe_stage.get("stage") == "failed":
                    state = "failed"
                elif read_control(recipes)["paused"]:
                    state = "paused"
                phase = self.training("train-" + name, model["title"], recipes / name,
                                      recipes / ("train-" + name + ".log"), state, model["architecture"])
                phase["note"] = recipe_plan["note"] + f' Model seed {model["seed"]}; split seed {recipe_plan["split_seed"]}.'
                phases.append(phase)
            stage, queue_alive, control = recipe_stage, recipe_alive, read_control(recipes)
        gpu = None
        gpu_files = [path for path in [self.activation / "gpu.csv", self.queue / "gpu.csv",
                                      pooling / "gpu.csv", comparisons / "gpu.csv", capacity / "gpu.csv",
                                      capacity / "overlap-battles/gpu.csv", third / "gpu.csv",
                                      recipes / "gpu.csv"] if path.exists()]
        if gpu_files:
            rows = tail(max(gpu_files, key=lambda path: path.stat().st_mtime), 4096)
            if rows:
                try:
                    row = next(csv.reader([rows[-1]], skipinitialspace=True))
                    gpu = {"timestamp": row[0], "utilization": float(row[2]),
                           "memory_mib": float(row[4]), "power_w": float(row[5]),
                           "temperature_c": float(row[8])}
                except (ValueError, IndexError):
                    pass
        return {"updated_at": now(), "phases": phases, "control": control,
                "worker": {"alive": queue_alive, **stage}, "activation_alive": activation_alive,
                "activation_failed": activation_failed, "selection": selection,
                "pooling_selection": pooling_selection, "gpu": gpu,
                "dataset": {key: dataset.get(key) for key in ("dataset_sha256", "training_games",
                    "training_positions", "validation_games", "validation_positions", "augmentation_count")},
                "settings": {"epochs": self.epochs, **future, "simulations": 4000,
                             "inference_batch_size": 64, "temperature": 0.7, "value_width": 64}}

    def dispatch(self, request):
        method = request.get("method")
        if method == 'game_image':
            from self_play_samples import read_game_image
            return read_game_image(self.queue, request.get('phase_id'), request.get('epoch'), request.get('sample'))
        if method == "snapshot":
            return apply_experiment_metadata(self.snapshot(), self.json(self.queue/'experiment-metadata.json'))
        if method == "logs":
            self.snapshot()
            identity = request.get("phase_id")
            if identity not in self.logs:
                raise ValueError("Unknown experiment")
            lines = [line for line in tail(self.logs[identity]) if "BATTLE_RESULT" not in line]
            return {"phase_id": identity, "lines": lines[-80:], "updated_at": now()}
        raise ValueError("Unknown RPC method")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--queue-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--games", type=int, default=300)
    args = parser.parse_args()
    agent_type = Experiments
    if (args.queue_dir / 'recipe-battle-plan.json').exists():
        from recipe_battle_agent import RecipeExperiments
        agent_type = RecipeExperiments
    experiments = agent_type(args.activation_dir, args.queue_dir, args.epochs, args.games)
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd) if os.isatty(fd) else None
    try:
        if previous is not None:
            tty.setraw(fd)
        print("ALZ_EXPERIMENT_AGENT_READY", flush=True)
        # The gateway can leave the container-side PTY open after disconnect.
        # Retire idle agents instead of accumulating abandoned SSH readers.
        while select.select([sys.stdin], [], [], 90)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            request = {}
            try:
                if len(line) > 4096:
                    raise ValueError("Request too large")
                request = json.loads(line)
                result = experiments.dispatch(request)
                response = {"id": request.get("id"), "result": result}
            except Exception as error:
                response = {"id": request.get("id"), "error": str(error)}
            print(json.dumps(response, allow_nan=False), flush=True)
    finally:
        if previous is not None:
            termios.tcsetattr(fd, termios.TCSANOW, previous)


if __name__ == "__main__":
    main()
