import argparse
import json

from common import latest_eval_log, metrics_path, parse_eval_log, write_metrics


COMPARE_DATASETS = {
    '4-17': 'mths',
    '4-18': 'dair',
    '4-19': 'v2x_real',
    '4-20': 'mths',
}


def infer_dataset(table, dataset):
    if dataset:
        return dataset
    return COMPARE_DATASETS.get(table, 'mths')


def main():
    parser = argparse.ArgumentParser(description='Rebuild exp46 AP summaries from existing eval logs.')
    parser.add_argument('--table', required=True)
    parser.add_argument('--row', required=True)
    parser.add_argument('--dataset', choices=['mths', 'dair', 'v2x_real'], default=None)
    args = parser.parse_args()

    dataset = infer_dataset(args.table, args.dataset)
    log_path = latest_eval_log(args.table, args.row)
    if log_path is None:
        raise FileNotFoundError(f'No log_eval_*.txt found for {args.table}/{args.row}')

    path = metrics_path(args.table, args.row)
    if path.exists():
        with open(path, 'r') as f:
            payload = json.load(f)
    else:
        payload = {'table': args.table, 'row': args.row, 'dataset': dataset}

    payload.update(parse_eval_log(log_path, dataset=dataset))
    payload.setdefault('table', args.table)
    payload.setdefault('row', args.row)
    payload.setdefault('dataset', dataset)
    payload['eval_log'] = str(log_path)
    out_path = write_metrics(args.table, args.row, payload)
    print(f'refreshed {out_path}')
    print(f'eval log: {log_path}')


if __name__ == '__main__':
    main()
