"""Summarize paired timings and preserve the completed regression result."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--test-log', default='/tmp/softplus_warp8_all_tests.log')
args = ap.parse_args()
root = Path(__file__).resolve().parent
summary = {'runs': {}}
for name in ('final', 'reverse'):
    path = root / f'softplus_warp8_{name}.json'
    rows = json.loads(path.read_text())
    result = []
    for row in rows:
        if row['phase'] != 'train':
            continue
        times = {k: v['ms'] for k, v in row['results'].items()}
        result.append({
            'shape': [row[k] for k in ('b', 't', 'h', 'd')],
            'ms': times,
            'softplus_time_reduction_pct': 100 * (1 - times['sp_after'] / times['sp_before']),
            'softplus_over_sm256_pct': 100 * (times['sp_after'] / times['sm_256'] - 1),
            'max_relative_to_peak_error': max(row['errors']),
        })
    summary['runs'][name] = result

log = Path(args.test_log).read_text()
match = re.search(r'Ran (\d+) tests in ([\d.]+)s', log)
if match is None or not log.rstrip().endswith('OK'):
    raise RuntimeError('Regression suite has not finished successfully; inspect the test log')
summary['tests'] = {'count': int(match[1]), 'seconds': float(match[2]), 'result': 'OK'}
snapshot = root / 'softplus_warp8_tests.log'
snapshot.write_text(log)
paths = [root.parent / p for p in (
    'flash_attn_4/flash_bwd.py', 'flash_attn_4/interface.py', 'flash_attn_4/softplus_api.py',
    'tests/test_softplus_warp8.py', 'trains/bench_softplus_warp8.py',
    'trains/softplus_warp8_final.json', 'trains/softplus_warp8_reverse.json',
    'trains/softplus_warp8_tests.log', 'trains/SOFTPLUS_WARP8_RESULTS.md',
)]
summary['final_hashes'] = {str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(root / 'softplus_warp8_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
