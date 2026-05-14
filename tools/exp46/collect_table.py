import argparse

from common import read_metrics


def fmt(value):
    if isinstance(value, float):
        return f'{value:.4f}'
    if value is None:
        return ''
    return str(value)


def main():
    parser = argparse.ArgumentParser(description='Collect Chapter 4.6 metrics into a Markdown table.')
    parser.add_argument('--table', required=True)
    args = parser.parse_args()

    rows = read_metrics(args.table)
    if not rows:
        print(f'No metrics found for table {args.table}')
        return
    if args.table == '4-21':
        columns = ['row', 'variant', 'dataset', 'params', 'trainable_params', 'elapsed_sec', 'profile_error']
    else:
        columns = ['row', 'method', 'variant', 'dataset', 'epochs', 'ap_mean', 'recall_rcnn_0.3', 'recall_rcnn_0.5', 'recall_rcnn_0.7']
    print('| ' + ' | '.join(columns) + ' |')
    print('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for row in rows:
        print('| ' + ' | '.join(fmt(row.get(col)) for col in columns) + ' |')


if __name__ == '__main__':
    main()
