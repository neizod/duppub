import sys
from contextlib import contextmanager
from io import StringIO
from report import process, parse_arguments

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def test_report():
    with captured_output() as (out, err):
        process(parse_arguments({'--algorithm': 'levenshtein',
                                 '--limit_chars': '100',
                                 '--threshold': '80',
                                 '<CSV>': 'example_input.csv'}))

    expected_output = ('|  score  |          id-1          |          id-2          |\n'
                       '|---------|------------------------|------------------------|\n'
                       '|  99.50% | cross-publisher-2      | cross-publisher-3      |\n'
                       '|  99.50% | cross-publisher-1      | cross-publisher-3      |\n'
                       '|  99.50% | cross-publisher-1      | cross-publisher-2      |\n'
                       '|  99.50% | arXiv-v3               | arXiv-v4               |\n'
                       '|  99.50% | arXiv-v1               | arXiv-v2               |')
    assert out.getvalue().strip() == expected_output

