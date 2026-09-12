#!/usr/bin/env python3
"""Assemble the standalone report with embedded histograms and design tokens."""
import argparse
import hashlib
from datetime import datetime, timezone
import html
import json
from pathlib import Path


def median(hist):
    n = sum(hist)
    def at(index):
        acc = 0
        for i, count in enumerate(hist):
            acc += count
            if acc > index:
                return i
    return (at((n-1)//2) + at(n//2))/2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    assets = Path(__file__).resolve().parent/'report_assets/nucleus'
    parser.add_argument("--tokens", type=Path, default=assets/'tokens.css')
    parser.add_argument("--plotly", type=Path, required=True, help='Local Plotly bundle matching report_assets/nucleus/plotly-4.0.0.min.source.json')
    parser.add_argument("--output", type=Path, required=True, help='Generated standalone HTML (usually under runs/)')
    args = parser.parse_args()
    root = args.report_dir
    source = json.loads((assets/'plotly-4.0.0.min.source.json').read_text())
    plotly = args.plotly.read_bytes()
    if hashlib.sha256(plotly).hexdigest() != source['sha256']:
        parser.error('Plotly bundle does not match the recorded checksum')
    data = json.loads((root/'distributions.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    sources = json.loads((root/'sources.json').read_text())
    rows = []
    for p in ['0.99', '0.98', '0.95', '0.9', '1.0']:
        groups = list(data['thresholds'][p].values())
        n = sum(g['positions'] for g in groups)
        before = sum(g['sum_before'] for g in groups)
        removed = sum(g['sum_removed'] for g in groups)
        hist = [sum(g['removed_count'][i] for g in groups) for i in range(362)]
        a = summary['thresholds'][p]
        cells = [f'{removed:,}', f'{removed/n:.1f}', f'{median(hist):g}', f'{removed/before:.2%}',
                 f'{a["removed_mass"]["mean"]:.2%}', f'{a["excluded_action_fraction"]:.2%}',
                 f'{a["games_with_excluded_observed_action_fraction"]:.2%}']
        rows.append(f'<tr class="{"focus-row" if p=="0.95" else ""}"><th scope="row">{p}</th>'+''.join(f'<td>{c}</td>' for c in cells)+'</tr>')
        if p == '0.95':
            mean_removed = f'{removed/n:.1f}'
            weighted_removed = f'{removed/before:.2%}'
    source_rows = ''.join(f'<tr><th scope="row">{s["checkpoint"]["epoch"]}</th><td class="hash">{html.escape(s["replay_sha256"])}</td></tr>' for s in sources['sources'])
    result = (assets/'report-template.html').read_text()
    for key,value in {'/*TOKENS*/':args.tokens.read_text(), '/*DATA*/':json.dumps(data,separators=(',',':')).replace('<','\\u003c'),
                      '/*PLOTLY*/':plotly.decode().replace('</script', '<\\/script'),
                      '/*REPORT_JS*/':(assets/'report-plots.js').read_text(),
                      '<!--GENERATED-->':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                      '<!--MEAN_REMOVED-->':mean_removed,'<!--WEIGHTED_REMOVED-->':weighted_removed,
                      '<!--COMPARISON_ROWS-->':''.join(rows),'<!--SOURCE_ROWS-->':source_rows}.items():
        assert key in result
        result = result.replace(key,value)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result)
    print(output.resolve())


if __name__=='__main__':
    main()
