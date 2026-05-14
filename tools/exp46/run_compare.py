import argparse
import time

from common import (
    COMPARE_METHODS,
    METHOD_FUSION_TYPE,
    METHOD_LABELS,
    METHOD_MODALITY,
    METHOD_VARIANTS,
    latest_eval_log,
    parse_eval_log,
    run_train,
    write_cfg,
    write_metrics,
)


def resolve_method(name):
    if name in METHOD_VARIANTS:
        return name
    for row, label in METHOD_LABELS.items():
        if name == label:
            return row
    raise KeyError(f'Unknown compare method: {name}')


def main():
    parser = argparse.ArgumentParser(description='Run Chapter 4.6 method comparison table.')
    parser.add_argument('--table', required=True)
    parser.add_argument('--dataset', default='mths', choices=['mths', 'dair', 'v2x_real'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--metric-set', default='main')
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=None, help='global batch size passed to tools/train.py')
    parser.add_argument('--max-voxels', type=int, default=None, help='cap train/test MAX_NUMBER_OF_VOXELS in generated config')
    parser.add_argument('--amp', action='store_true', help='enable mixed precision training')
    args = parser.parse_args()

    methods = [resolve_method(name) for name in args.only] if args.only else COMPARE_METHODS
    for method_key in methods:
        variant = METHOD_VARIANTS[method_key]
        row = method_key
        method_label = METHOD_LABELS.get(method_key, method_key)
        cfg_path = write_cfg(
            variant, args.dataset, args.epochs,
            f'{args.table}_{method_key}_{args.dataset}',
            max_voxels=args.max_voxels,
        )
        start = time.time()
        cmd = run_train(
            cfg_path, args.table, row, args.epochs,
            workers=args.workers,
            batch_size=args.batch_size,
            use_amp=args.amp,
        )
        metrics = parse_eval_log(latest_eval_log(args.table, row), dataset=args.dataset)
        metrics.update({
            'table': args.table,
            'row': row,
            'method': method_label,
            'modality': METHOD_MODALITY.get(method_key),
            'fusion_type': METHOD_FUSION_TYPE.get(method_key),
            'dataset': args.dataset,
            'variant': variant,
            'metric_set': args.metric_set,
            'epochs': args.epochs,
            'config': str(cfg_path),
            'train_cmd': cmd,
            'elapsed_sec': time.time() - start,
        })
        path = write_metrics(args.table, row, metrics)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
