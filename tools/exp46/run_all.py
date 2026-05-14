import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

from common import EXP_DIR, ROOT_DIR


ABLATION_TABLES = [
    ('4-10', ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']),
    ('4-11', ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']),
    ('4-12', ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']),
    ('4-13', ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']),
    ('4-14', ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']),
    ('4-15', ['F1', 'F2', 'F3', 'F4', 'F5']),
    ('4-16', ['G1', 'G2', 'G3', 'G4']),
]

COMPARE_TABLES = [
    ('4-17', 'mths', 'main'),
    ('4-18', 'dair', 'main'),
    ('4-19', 'v2x_real', 'main'),
    ('4-20', 'mths', 'class_ap'),
]

PROFILE_ROWS = ['voxelnext', 'v43', 'v43_v45', 'det2d', 'full']


def now():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')


def stream_command(name, cmd, log_path, status_path, continue_on_error):
    started = time.time()
    banner = f'\n[{now()}] START {name}\n$ {" ".join(cmd)}\n'
    print(banner, flush=True)
    with open(log_path, 'a') as log_file:
        log_file.write(banner)
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='', flush=True)
            log_file.write(line)
            log_file.flush()
        ret = proc.wait()
        elapsed = time.time() - started
        footer = f'\n[{now()}] END {name} rc={ret} elapsed_sec={elapsed:.1f}\n'
        print(footer, flush=True)
        log_file.write(footer)

    append_jsonl(status_path, {
        'name': name,
        'cmd': cmd,
        'returncode': ret,
        'elapsed_sec': elapsed,
        'started_at': started,
        'ended_at': time.time(),
    })
    if ret != 0 and not continue_on_error:
        raise SystemExit(ret)
    return ret


def collect_table(table, log_path, status_path, continue_on_error):
    out_path = EXP_DIR / table / 'table.md'
    cmd = [sys.executable, 'tools/exp46/collect_table.py', '--table', table]
    started = time.time()
    name = f'collect {table}'
    print(f'\n[{now()}] START {name}', flush=True)
    proc = subprocess.run(cmd, cwd=ROOT_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, end='', flush=True)
    with open(log_path, 'a') as log_file:
        log_file.write(f'\n[{now()}] START {name}\n$ {" ".join(cmd)}\n')
        log_file.write(proc.stdout)
        log_file.write(f'\n[{now()}] END {name} rc={proc.returncode} output={out_path}\n')
    append_jsonl(status_path, {
        'name': name,
        'cmd': cmd,
        'returncode': proc.returncode,
        'elapsed_sec': time.time() - started,
        'output': str(out_path),
    })
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description='Run all Chapter 4.6 experiments in sequence.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--skip-check-data', action='store_true')
    args = parser.parse_args()

    run_id = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = EXP_DIR / '_run_all'
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f'run_all_{run_id}.log'
    status_path = run_dir / f'status_{run_id}.jsonl'

    print(f'run_id={run_id}')
    print(f'log={log_path}')
    print(f'status={status_path}')

    if not args.skip_check_data:
        stream_command(
            'check_data',
            [sys.executable, 'tools/exp46/check_data.py'],
            log_path,
            status_path,
            args.continue_on_error,
        )

    for table, rows in ABLATION_TABLES:
        for row in rows:
            stream_command(
                f'{table} {row}',
                [
                    sys.executable, 'tools/exp46/run_row.py',
                    '--table', table,
                    '--row', row,
                    '--dataset', 'mths',
                    '--epochs', str(args.epochs),
                    '--workers', str(args.workers),
                ],
                log_path,
                status_path,
                args.continue_on_error,
            )
        collect_table(table, log_path, status_path, args.continue_on_error)

    for table, dataset, metric_set in COMPARE_TABLES:
        cmd = [
            sys.executable, 'tools/exp46/run_compare.py',
            '--table', table,
            '--dataset', dataset,
            '--epochs', str(args.epochs),
            '--metric-set', metric_set,
            '--workers', str(args.workers),
        ]
        stream_command(f'{table} compare {dataset}', cmd, log_path, status_path, args.continue_on_error)
        collect_table(table, log_path, status_path, args.continue_on_error)

    for row in PROFILE_ROWS:
        stream_command(
            f'4-21 {row}',
            [sys.executable, 'tools/exp46/profile_row.py', '--row', row, '--dataset', 'mths'],
            log_path,
            status_path,
            args.continue_on_error,
        )
    collect_table('4-21', log_path, status_path, args.continue_on_error)

    print(f'\n[{now()}] all requested experiments finished')
    print(f'log: {log_path}')
    print(f'status: {status_path}')
    print(f'table markdown files: {EXP_DIR}/<table>/table.md')


if __name__ == '__main__':
    main()
