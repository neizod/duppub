#!/usr/bin/env python3

import sys
import csv
from enum import IntEnum


# OBSERVATION:
# - not many duplicate more than 3 records...


class Field(IntEnum):
    ID       = 0
    AUTHOR   = 1
    NAME     = 2
    YEAR     = 3
    ABSTRACT = 4


def edit_distant(this_str, other_str):
    if len(this_str) > len(other_str):
        this_str, other_str = other_str, this_str
    distant = list(range(1 + len(this_str)))
    for i2, c2 in enumerate(other_str):
        new_distant = [i2 + 1]
        for i1, c1 in enumerate(this_str):
            if c1 == c2:
                new_distant += [distant[i1]]
            else:
                new_distant += [1 + min(distant[i1], distant[i1+1], new_distant[-1])]
        distant = new_distant
    return distant[-1]


def difference(this_row, other_row, limit_char=50):
    if not limit_char:
        this_abstract = this_row[Field.ABSTRACT]
        other_abstract = other_row[Field.ABSTRACT]
    else:
        this_abstract = this_row[Field.ABSTRACT][:limit_char]
        other_abstract = other_row[Field.ABSTRACT][:limit_char]
    num_chars = max(len(this_abstract), len(other_abstract))
    if not num_chars:
        return 0
    distant = edit_distant(this_abstract, other_abstract)
    return 100 - (100 * distant / num_chars)


def read_as_records(filename):
    return {row[Field.ID]: row for row in csv.reader(open(filename))}


def process(filename):
    records = read_as_records(filename)
    table = {this_id: {other_id: 0 for other_id in records} for this_id in records}
    for this_id, this_row in records.items():
        this_row[Field.ABSTRACT]
        for other_id, other_row in records.items():
            if this_id == other_id:
                continue
            if this_id > other_id:
                continue
            table[this_id][other_id] = difference(this_row, other_row)
    report(table)


def report(table, percent=80):
    output = []
    for this_id, others in table.items():
        for other_id, similarity_rate in others.items():
            if this_id == other_id:
                continue
            if this_id > other_id:
                continue
            output += [(similarity_rate, this_id, other_id)]
    for line in filter(lambda x: x[0] >= 80, reversed(sorted(output))):
        print(f'{line[1]} {line[2]}: {line[0]:.2f}%')


def main():
    if len(sys.argv) == 1:
        print('Please add a CSV file to detect duplications.')
        raise IOError('no file')
    if not sys.argv[1].endswith('.csv'):
        print('File must be CSV.')
        raise IOError('file not csv')
    if len(sys.argv) >= 3:
        print('Too many CSV files')
        raise IOError('too many files')
    filename = sys.argv[1]
    process(filename)


if __name__ == '__main__':
    main()
