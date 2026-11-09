#!/usr/bin/python3
"""
Concatenate the per-signal peak CSVs into one table.

Reads the ``<stem>_bl_<mol>.csv`` files that ``tools/loop_extract.sh``
writes, one per signal and molecule, and writes them as one table to
``solvent_effect_data.csv`` in the current directory. An ``id`` column,
the replicate number taken from the file name (``<...>__<id>_<...>.csv``),
is inserted as the third column.

Usage
-----
python tools/data_aggregator.py FILE.csv [FILE.csv ...]
"""

import argparse
import os
import re

import pandas as pd


def main() -> None:
    """CLI entry point of the peak-table aggregation."""
    parser = argparse.ArgumentParser(
        prog='data_aggregator',
        description='concatenate peak CSVs into solvent_effect_data.csv')
    parser.add_argument('filename', nargs='+', help='the .csv input files')
    args = parser.parse_args()

    chromato_list = []
    for path in args.filename:
        filename = os.path.basename(path)
        chromato = pd.read_csv(path)
        chromato['id'] = re.match(r'^.+__(\d+)_\w+.csv', filename).group(1)
        chromato_list.append(chromato)

    df = pd.concat(chromato_list, ignore_index=True)
    first_col = df.pop('id')
    df.insert(2, 'id', first_col)
    df.to_csv('solvent_effect_data.csv', index=False)


if __name__ == '__main__':
    main()
