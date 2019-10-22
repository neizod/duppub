#!/usr/bin/env python3

import argparse
import csv
import math
import re
from collections import Counter
from enum import IntEnum

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
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost =2
            distance[row][col] = min( distance[row-1][col] + 1,
                                      distance[row][col-1] + 1,
                                      distance[row-1][col-1] + cost )
    ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
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
                new_distant += [1 + min(distance[i1], distance[i1+1], new_distant[-1])]
        distance = new_distant
    return 100 - (100 * distance[-1] / max(len(this_str), len(other_str)))


def cosine_similarity_ratio(this_str: str, other_str: str) -> float:
    """
    calculate cosine similarity between two strings
    :param this_str:
    :param other_str:
    :return:
    """
    def get_cosine(vec1: Counter, vec2: Counter) ->float:
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
            table[this_id][other_id] = difference( this_row, other_row,
                                                   limit_chars=arguments.limit_chars,
                                                   distance_algorithm=arguments.algorithm )
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


def parse_arguments():
    algorithms = {
        'levenshtein': levenshtein_ratio,
        'edit_distance': edit_distance_ratio,
        'cosine_distance': cosine_similarity_ratio
    }
    parser = argparse.ArgumentParser(description='DupPub detects duplicate publications')
    parser.add_argument('csv', type=str,  help='CSV file to process')
    parser.add_argument('--threshold', type=int, default=80, help='Threshold percentage, as an integer.')
    parser.add_argument('--limit_chars', type=int, default=100, help='String length limit to increase performance.')
    parser.add_argument('--algorithm', type=str, default='levenshtein', help='The algorithm to use.')
    arguments = parser.parse_args()
    if not 0 <= arguments.threshold <= 100:
        parser.error('THRESHOLD not in range 0% to 100%')
    if arguments.algorithm in algorithms.keys():
        arguments.algorithm = algorithms[arguments.algorithm]
    else:
        parser.error('ALGORITHM not exists')
    return arguments


def main():
    process(parse_arguments())


if __name__ == '__main__':
    main()
