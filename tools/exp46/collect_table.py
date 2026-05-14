import argparse

from common import build_table_markdown, read_metrics, write_table_markdown


def main():
    parser = argparse.ArgumentParser(description='Collect Chapter 4.6 metrics into a Markdown table.')
    parser.add_argument('--table', required=True)
    args = parser.parse_args()

    rows = read_metrics(args.table)
    if not rows:
        print(f'No metrics found for table {args.table}')
        return
    markdown = build_table_markdown(args.table, rows)
    table_path = write_table_markdown(args.table)
    print(markdown)
    print(f'\nwrote {table_path}')


if __name__ == '__main__':
    main()
