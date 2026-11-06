"""Dashboard summaries from relevant structured events, independent of journal paging."""

from collections import Counter
import math


def distribution(values):
    if not values:
        return None
    values = sorted(values)

    def quantile(p):
        index = p * (len(values) - 1)
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "count": len(values),
        "minimum": values[0],
        "maximum": values[-1],
        "mean": sum(values) / len(values),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "histogram": [],
        "frequencies": [
            {"moves": n, "count": count} for n, count in sorted(Counter(values).items())
        ],
    }


def summarize(events):
    games, finishes = {}, {}
    started = None
    saved = benchmark = participants = None
    completed = False
    for event in events:
        kind, payload = event["kind"], event["payload"]
        if kind == "worker_started" and started is None:
            started = event["time"]
        elif kind == "comparison_game":
            games[payload["game"]] = payload
            if not payload.get("finish_time_unavailable"):
                finishes.setdefault(payload["game"], event["time"])
        elif kind == "comparison_completed":
            saved = payload.get("game_statistics") or saved
            completed = True
        elif kind == "benchmark_completed":
            benchmark = payload
        elif kind == "imported":
            participants = payload.get("participants") or participants
    result = {"comparison": saved, "benchmark": benchmark, "participants": participants}
    # Imported aggregates can cover more games than the retained raw archive.
    if not games or (saved and saved.get("games", 0) > len(games)):
        return result
    sides = ("first_checkpoint", "second_checkpoint")
    seats = {
        side: {
            seat: dict(games=0, wins=0, draws=0, losses=0)
            for seat in ("first", "second")
        }
        for side in sides
    }
    outcomes = dict(first_player_wins=0, second_player_wins=0, draws=0)
    for game in games.values():
        winner = game.get("winner")
        for side in sides:
            count = seats[side]["first" if game["first_seat"] == side else "second"]
            count["games"] += 1
            count[
                "draws" if not winner else "wins" if winner == side else "losses"
            ] += 1
        outcomes[
            "draws"
            if not winner
            else "first_player_wins"
            if winner == game["first_seat"]
            else "second_player_wins"
        ] += 1
    lengths = {"all": distribution([g["plies"] for g in games.values()])}
    for side in (*sides, "draws"):
        lengths[side] = distribution(
            [g["plies"] for g in games.values() if (g.get("winner") or "draws") == side]
        )
    completion = (saved or {}).get("completion")
    if started is not None and completed and len(finishes) == len(games):
        times = sorted(max(0, t - started) for t in finishes.values())
        completion = {
            f"p{p}_seconds": times[max(0, math.ceil(len(times) * p / 100) - 1)]
            for p in (50, 90, 95, 99)
        }
        completion |= {
            "last_finish_seconds": times[-1],
            "final_10_percent_seconds": times[-1] - completion["p90_seconds"],
            "includes_pauses": True,
            "points": [{"games": i + 1, "seconds": t} for i, t in enumerate(times)],
        }
    result["comparison"] = {
        "games": len(games),
        "seat_results": seats,
        "outcomes": outcomes,
        "lengths": lengths,
        "durations": distribution(
            [
                g["duration_seconds"]
                for g in games.values()
                if isinstance(g.get("duration_seconds"), (int, float))
                and math.isfinite(g["duration_seconds"])
            ]
        ),
        "completion": completion,
    }
    return result
