"""Read-only summaries of completed match reports and their progress logs."""
import math
from collections import Counter
from pathlib import Path
import re
import statistics


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    def percentile(fraction):
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]
    buckets = []
    for lower in range(0, (ordered[-1] // 20 + 1) * 20, 20):
        buckets.append({'lower': lower, 'upper': lower + 19,
                        'count': sum(lower <= value < lower + 20 for value in ordered)})
    return {'count': len(ordered), 'minimum': ordered[0], 'maximum': ordered[-1],
            'mean': statistics.fmean(ordered), 'median': statistics.median(ordered),
            'p90': percentile(.90), 'p95': percentile(.95), 'p99': percentile(.99),
            'histogram': buckets,
            'frequencies': [{'moves': moves, 'count': count} for moves, count in sorted(Counter(ordered).items())]}


def completion_timing(log, games, duration):
    """Series-relative finish times, never estimates of individual game durations."""
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        return None
    pattern = re.compile(r'games_completed=(\d+).*games_total=(\d+).*elapsed_seconds=([\d.]+)')
    points = [{'games': 0, 'seconds': 0.0}]
    previous = (0, 0.0)
    try:
        with Path(log).open() as stream:
            for line in stream:
                match = pattern.search(line)
                if not match:
                    continue
                count, total, elapsed = int(match[1]), int(match[2]), float(match[3])
                if total != games or not 0 <= count <= games or not math.isfinite(elapsed):
                    continue
                # Logs can append attempts; only the final complete attempt is relevant.
                if count < previous[0] or elapsed < previous[1]:
                    points = [{'games': 0, 'seconds': 0.0}]
                if count > points[-1]['games']:
                    points.append({'games': count, 'seconds': elapsed})
                previous = count, elapsed
    except FileNotFoundError:
        return None
    if points[-1]['games'] != games or abs(points[-1]['seconds'] - duration) > max(2, duration * .001):
        return None
    milestones = {}
    keep = {0, len(points) - 1}
    for percentage in (50, 90, 95, 99):
        threshold = math.ceil(games * percentage / 100)
        index = next(i for i, point in enumerate(points) if point['games'] >= threshold)
        keep.add(index)
        milestones[f'p{percentage}_seconds'] = points[index]['seconds']
    stride = max(1, math.ceil(len(points) / 300))
    keep.update(range(0, len(points), stride))
    return {**milestones, 'first_finish_seconds': points[1]['seconds'],
            'last_finish_seconds': points[-1]['seconds'],
            'final_10_percent_seconds': max(0, points[-1]['seconds'] - milestones['p90_seconds']),
            'points': [points[i] for i in sorted(keep)]}


def summarize_match(report, log):
    games = report.get('games', [])
    competitors = ('first_checkpoint', 'second_checkpoint')
    if not games or any(type(g.get('plies')) is not int or g['plies'] < 0 or
                        g.get('first_seat') not in competitors or g.get('second_seat') not in competitors or
                        g['first_seat'] == g['second_seat'] or 'winner' not in g or g['winner'] not in (*competitors, None)
                        for g in games):
        return None
    lengths = {'all': [], 'first_checkpoint': [], 'second_checkpoint': [], 'draws': []}
    seats = {competitor: {seat: {'games': 0, 'wins': 0, 'draws': 0, 'losses': 0}
                         for seat in ('first', 'second')} for competitor in competitors}
    outcomes = {'first_player_wins': 0, 'second_player_wins': 0, 'draws': 0}
    for game in games:
        winner = game['winner']
        lengths['all'].append(game['plies'])
        lengths[winner or 'draws'].append(game['plies'])
        outcome = 'draws' if winner is None else 'first_player_wins' if winner == game['first_seat'] else 'second_player_wins'
        outcomes[outcome] += 1
        for seat in ('first', 'second'):
            competitor = game[seat + '_seat']
            totals = seats[competitor][seat]
            totals['games'] += 1
            totals['draws' if winner is None else 'wins' if winner == competitor else 'losses'] += 1
    return {'games': len(games), 'seat_results': seats, 'outcomes': outcomes,
            'lengths': {key: distribution(values) for key, values in lengths.items()},
            'completion': completion_timing(log, len(games), report.get('duration_seconds')),
            'individual_game_durations_available': False}


def summarize_live_match(log, planned_games):
    """Summarize only complete game events from the current log attempt."""
    games = {}
    competitors = ('first_checkpoint', 'second_checkpoint')
    previous = (0, 0.0)
    pattern = re.compile(r'games_completed=(\d+).*games_total=(\d+).*elapsed_seconds=([\d.]+)')
    try:
        with Path(log).open() as stream:
            for line in stream:
                if not line.endswith('\n'):
                    continue  # A writer may currently be publishing this event.
                if 'using compute device' in line:
                    games.clear()
                    previous = (0, 0.0)
                progress = pattern.search(line)
                if progress:
                    count, total, elapsed = int(progress[1]), int(progress[2]), float(progress[3])
                    if total != planned_games:
                        games.clear()
                        continue
                    if count < previous[0] or elapsed < previous[1]:
                        games.clear()
                    previous = (count, elapsed)
                if 'battle game complete' not in line:
                    continue
                fields = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
                fields = {key:value.strip('"') for key,value in fields.items()}
                try:
                    game, plies = int(fields['game']), int(fields['plies'])
                    first, second, winner = (fields[k] for k in ('first_seat','second_seat','winner'))
                except (KeyError, ValueError):
                    continue
                if not (0 <= game < planned_games and 0 <= plies <= 361 and
                        first in competitors and second in competitors and first != second and
                        winner in (*competitors,'draw')):
                    continue
                record = {'first_seat':first,'second_seat':second,
                          'winner':None if winner=='draw' else winner,'plies':plies}
                if game in games and games[game] != record:
                    games.clear()  # A repeated ID with a different result is a new attempt.
                games[game] = record
    except FileNotFoundError:
        return None
    summary = summarize_match({'games':list(games.values())}, log)
    if summary:
        summary.update(partial=True, planned_games=planned_games)
    return summary
