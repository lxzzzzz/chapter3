import argparse
import time

from common import ROW_VARIANTS, latest_eval_log, parse_eval_log, run_train, write_cfg, write_metrics


def main():
    parser = argparse.ArgumentParser(description='Run one Chapter 4.6 ablation table row.')
    parser.add_argument('--table', required=True)
    parser.add_argument('--row', required=True)
    parser.add_argument('--dataset', default='mths', choices=['mths', 'dair', 'v2x_real'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    variant = ROW_VARIANTS[args.row]
    cfg_path = write_cfg(variant, args.dataset, args.epochs, f'{args.table}_{args.row}_{args.dataset}')
    start = time.time()
    cmd = run_train(cfg_path, args.table, args.row, args.epochs, workers=args.workers)
    metrics = parse_eval_log(latest_eval_log())
    metrics.update({
        'table': args.table,
        'row': args.row,
        'dataset': args.dataset,
        'variant': variant,
        'epochs': args.epochs,
        'config': str(cfg_path),
        'train_cmd': cmd,
        'elapsed_sec': time.time() - start,
    })
    path = write_metrics(args.table, args.row, metrics)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
