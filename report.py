#!/usr/bin/env python3

"""Usage:
  report.py [-t N] [-l N] [--algorithm S] <CSV>

Detect duplicate or similar publications from database. This project aim to reduce size of the database by showing pairs of suspect duplications, to help citation easier and cleaner.

Export database as CSV file without header, with these fields: ID, Authors, Title of Article, Year, Abstract

Arguments:
  CSV                      Input CSV file to process

Options:
  -h --help                show this help message and exit
  -v, --version            show version and exit
  -t, --threshold N        Reporting threshold percentage, as an integer [default: 80]
  -l, --limit_chars N      String length limit. Decrease number to increase performance and degrade accuracy, as an integer [default: 100]
  --algorithm S            Algorithms: levenshtein, edit_distance, cosine_similarity, as a string [default: levenshtein]

Try:
  ./report.py publications.csv
  ./report.py -t 10 publications.csv
  ./report.py -t 10 -l 100 publications.csv
  ./report.py -t 10 -l 100 --algorithm edit_distance publications.csv
"""

import csv
import math
import re
from collections import Counter
from enum import IntEnum

from docopt import docopt
import numpy as np


# OBSERVATION:
# - not many duplicate more than 3 records...


class Field(IntEnum):
    ID       = 0
    AUTHOR   = 1
    NAME     = 2
    YEAR     = 3
    ABSTRACT = 4


def levenshtein_ratio(s, t):
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)
    distance[0, :] = np.arange(1, cols + 1)
    distance[:, 0] = np.arange(1, rows + 1)
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 2
            distance[row][col] = min(distance[row - 1][col] + 1,
                                     distance[row][col - 1] + 1,
                                     distance[row - 1][col - 1] + cost)
    ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
    return 100 * ratio


def edit_distance_ratio(this_str, other_str):
    if len(this_str) > len(other_str):
        this_str, other_str = other_str, this_str
    distance = list(range(1 + len(this_str)))
    for i2, c2 in enumerate(other_str):
        new_distant = [i2 + 1]
        for i1, c1 in enumerate(this_str):
            if c1 == c2:
                new_distant += [distance[i1]]
            else:
                new_distant += [1 + min(distance[i1], distance[i1 + 1], new_distant[-1])]
        distance = new_distant
    return 100 - (100 * distance[-1] / max(len(this_str), len(other_str)))


def cosine_similarity_ratio(this_str: str, other_str: str) -> float:
    """
    calculate cosine similarity between two strings
    :param this_str:
    :param other_str:
    :return:
    """
    def get_cosine(vec1: Counter, vec2: Counter) -> float:
        """
        calculate cosine similarity
        :param vec1:
        :param vec2:
        :return:
        """
        numerator = sum([vec1[x] * vec2[x] for x in set(vec1.keys() & vec2.keys())])
        denominator = math.sqrt(sum([vec1[x]**2 for x in vec1.keys()])) * math.sqrt(sum([vec2[x]**2 for x in vec2.keys()]))

        return float(numerator) / denominator if denominator else 0.0

    def text_to_vector(text):
        """
        change text to vector(BOW)
        :param text:
        :return:
        """
        return Counter(re.compile(r'\w+').findall(text))

    return 100 * get_cosine(text_to_vector(this_str), text_to_vector(other_str))


def difference(this_row, other_row, limit_chars=100, distance_algorithm=levenshtein_ratio):
    if not limit_chars:
        this_abstract = this_row[Field.ABSTRACT]
        other_abstract = other_row[Field.ABSTRACT]
    else:
        this_abstract = this_row[Field.ABSTRACT][:limit_chars]
        other_abstract = other_row[Field.ABSTRACT][:limit_chars]
    num_chars = max(len(this_abstract), len(other_abstract))
    if not num_chars:
        return 0
    return distance_algorithm(this_abstract, other_abstract)


def read_as_records(filename):
    return {row[Field.ID]: row for row in csv.reader(open(filename))}


def process(arguments):
    records = read_as_records(arguments.csv)
    table = {this_id: {other_id: 0 for other_id in records} for this_id in records}
    for this_id, this_row in records.items():
        this_row[Field.ABSTRACT]
        for other_id, other_row in records.items():
            if this_id == other_id:
                continue
            if this_id > other_id:
                continue
            table[this_id][other_id] = difference(this_row, other_row,
                                                  limit_chars=arguments.limit_chars,
                                                  distance_algorithm=arguments.algorithm)
    report(table, arguments.threshold)


def report(table, percent=80):
    output = []
    for this_id, others in table.items():
        for other_id, similarity_rate in others.items():
            if this_id == other_id:
                continue
            if this_id > other_id:
                continue
            output += [(similarity_rate, this_id, other_id)]
    print('|  score  |          id-1          |          id-2          |')
    print('|---------|------------------------|------------------------|')
    for line in filter(lambda x: x[0] >= percent, reversed(sorted(output))):
        print(f'| {line[0]:>6.2f}% | {line[1]:<22} | {line[2]:<22} |')


def merge(dict_1, dict_2):
    """Merge two dictionaries.
    Values that evaluate to true take priority over falsy values.
    `dict_1` takes priority over `dict_2`.
    """
    return dict((str(key).replace('<CSV>', 'csv').replace('--', ''),
                dict_1.get(key) or dict_2.get(key)) for key in set(dict_2) | set(dict_1))


class Struct:
    """Dummy class to construct objects from dictionaries."""
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_arguments(inputs):
    # Implemented Algorithms
    algorithms = {
        'levenshtein': levenshtein_ratio,
        'edit_distance': edit_distance_ratio,
        'cosine_distance': cosine_similarity_ratio
    }
    # Fallback values for input
    defaults = {
        '--threshold': 80,
        '--limit_chars': 100,
        '--algorithm': 'levenshtein'

    }
    # merge values together with defaults as fallback
    args = merge(inputs, defaults)
    arguments = Struct(**args)
    # recast and verify arguments integrity
    try:
        arguments.threshold = int(arguments.threshold)
        arguments.limit_chars = int(arguments.limit_chars)
        assert arguments.threshold <= 100 and arguments.threshold > 0
        assert arguments.algorithm in algorithms.keys()
    except Exception:
        raise ValueError("Invalid arguments either out of range or not found in list.")
    arguments.algorithm = algorithms[arguments.algorithm]
    return arguments


def main():
    """Parse input arguments."""
    inputs = docopt(__doc__, version='0.0.1')

    process(parse_arguments(inputs))


if __name__ == '__main__':
    main()

# end of code
