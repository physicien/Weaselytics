#!/usr/bin/python3
"""
Print a selection of the peak CSVs.

Concatenates the given CSVs and prints the rows of one molecule and the
chosen columns, every row and every column by default.

Usage
-----
python tools/viz_data.py FILE.csv [FILE.csv ...] [--mol MOL]
    [--columns COL [COL ...]]

For example, the retention times of ``t0`` by solvent:
``python tools/viz_data.py *.csv --mol t0 --columns solvent x0``.
"""

import argparse

import pandas as pd


def main() -> None:
    """CLI entry point of the peak-table viewer."""
    parser = argparse.ArgumentParser(
        prog='viz_data',
        description='print a selection of the peak CSVs')
    parser.add_argument('filename', nargs='+', help='the .csv input files')
    parser.add_argument('-m', '--mol', default=None,
                        help='keep only the rows of this molecule '
                             '(default: every row)')
    parser.add_argument('-c', '--columns', nargs='+', default=None,
                        help='print only these columns '
                             '(default: every column)')
    args = parser.parse_args()

    pd.set_option('display.max_rows', None)
    df = pd.concat([pd.read_csv(path) for path in args.filename],
                   ignore_index=True)

    if args.mol is not None:
        if args.mol not in set(df['mol']):
            parser.error(f'no molecule {args.mol!r} in the data')
        df = df[df['mol'] == args.mol]
    if args.columns is not None:
        missing = [c for c in args.columns if c not in df.columns]
        if missing:
            parser.error(f'no column {", ".join(missing)} in the data')
        df = df[args.columns]
    print(df)


if __name__ == '__main__':
    main()
