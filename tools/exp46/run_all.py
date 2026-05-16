import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

from common import COMPARE_METHODS, EXP_DIR, ROOT_DIR


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
]

PROFILE_ROWS = ['voxelnext', 'v43', 'v43_v45', 'det2d', 'full']

TABLE_ORDER = [table for table, _ in ABLATION_TABLES] + [table for table, _, _ in COMPARE_TABLES] + ['4-21']


def now():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')


def result_complete(table, row):
    row_dir = EXP_DIR / table / row
    if table == '4-21':
        return (row_dir / 'metrics.json').exists()
    if not all((row_dir / name).exists() for name in ('metrics.json', 'eval_summary.txt', 'table_values.json')):
        return False
    if table == '4-12':
        try:
            with open(row_dir / 'metrics.json', 'r') as f:
                return 'det2d_map_0.5' in json.load(f)
        except Exception:
            return False
    return True


def skip_command(name, status_path):
    print(f'\n[{now()}] SKIP {name} result files exist', flush=True)
    append_jsonl(status_path, {
        'name': name,
        'returncode': 0,
        'skipped': True,
        'reason': 'result files exist',
        'ended_at': time.time(),
    })


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


def should_run_table(table, start_table):
    if start_table is None:
        return True
    if start_table not in TABLE_ORDER:
        raise ValueError(f'Unknown --start-table {start_table}; choose one of: {", ".join(TABLE_ORDER)}')
    return TABLE_ORDER.index(table) >= TABLE_ORDER.index(start_table)


def main():
    parser = argparse.ArgumentParser(description='Run all Chapter 4.6 experiments in sequence.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=None, help='global batch size passed to every training job')
    parser.add_argument('--max-voxels', type=int, default=None, help='cap train/test MAX_NUMBER_OF_VOXELS in generated configs')
    parser.add_argument('--amp', action='store_true', help='enable mixed precision training')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--skip-check-data', action='store_true')
    parser.add_argument('--skip-existing', action='store_true', help='skip rows that already have complete result files')
    parser.add_argument('--start-table', default=None, help=f'start from this table, choices: {", ".join(TABLE_ORDER)}')
    args = parser.parse_args()

    if args.start_table is not None and args.start_table not in TABLE_ORDER:
        raise SystemExit(f'Unknown --start-table {args.start_table}; choose one of: {", ".join(TABLE_ORDER)}')

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
        if not should_run_table(table, args.start_table):
            print(f'\n[{now()}] SKIP {table} before start-table {args.start_table}', flush=True)
            continue
        for row in rows:
            if args.skip_existing and result_complete(table, row):
                skip_command(f'{table} {row}', status_path)
                continue
            stream_command(
                f'{table} {row}',
                [
                    sys.executable, 'tools/exp46/run_row.py',
                    '--table', table,
                    '--row', row,
                    '--dataset', 'mths',
                    '--epochs', str(args.epochs),
                    '--workers', str(args.workers),
                ]
                + ([] if args.batch_size is None else ['--batch-size', str(args.batch_size)])
                + ([] if args.max_voxels is None else ['--max-voxels', str(args.max_voxels)])
                + ([] if not args.amp else ['--amp']),
                log_path,
                status_path,
                args.continue_on_error,
            )
        collect_table(table, log_path, status_path, args.continue_on_error)

    for table, dataset, metric_set in COMPARE_TABLES:
        if not should_run_table(table, args.start_table):
            print(f'\n[{now()}] SKIP {table} before start-table {args.start_table}', flush=True)
            continue
        methods = [row for row in COMPARE_METHODS if not (args.skip_existing and result_complete(table, row))]
        if not methods:
            print(f'\n[{now()}] SKIP {table} compare {dataset} all metrics.json files exist', flush=True)
            collect_table(table, log_path, status_path, args.continue_on_error)
            continue
        cmd = [
            sys.executable, 'tools/exp46/run_compare.py',
            '--table', table,
            '--dataset', dataset,
            '--epochs', str(args.epochs),
            '--metric-set', metric_set,
            '--workers', str(args.workers),
        ]
        if args.batch_size is not None:
            cmd.extend(['--batch-size', str(args.batch_size)])
        if args.max_voxels is not None:
            cmd.extend(['--max-voxels', str(args.max_voxels)])
        if args.amp:
            cmd.append('--amp')
        if args.continue_on_error:
            cmd.append('--continue-on-error')
        if args.skip_existing:
            cmd.extend(['--only'] + methods)
        stream_command(f'{table} compare {dataset}', cmd, log_path, status_path, args.continue_on_error)
        collect_table(table, log_path, status_path, args.continue_on_error)

    if not should_run_table('4-21', args.start_table):
        print(f'\n[{now()}] SKIP 4-21 before start-table {args.start_table}', flush=True)
        print(f'\n[{now()}] all requested experiments finished')
        print(f'log: {log_path}')
        print(f'status: {status_path}')
        print(f'table markdown files: {EXP_DIR}/<table>/table.md')
        return

    for row in PROFILE_ROWS:
        if args.skip_existing and result_complete('4-21', row):
            skip_command(f'4-21 {row}', status_path)
            continue
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
