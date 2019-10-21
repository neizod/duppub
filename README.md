DupPub (Duplicate Publication)
==============================

Detect duplicate or similar publications from database. This project aim to reduce size of the database by showing pairs of suspect duplications, to help citation easier and cleaner.


Usage
-----

Export database as CSV file without header, with these fields:

1. ID
2. Authors
3. Title of the article
4. Year
5. Abstract

For example, if your exported CSV named `publications.csv`, then run it with:

    python3 report.py publications.csv


Example Result
--------------

From `example_input.csv`, this is the result:

    |  score  |          id-1          |          id-2          |
    |---------|------------------------|------------------------|
    | 100.00% | cross-publisher-2      | cross-publisher-3      |
    | 100.00% | cross-publisher-1      | cross-publisher-3      |
    | 100.00% | cross-publisher-1      | cross-publisher-2      |
    | 100.00% | arXiv-v3               | arXiv-v4               |
    | 100.00% | arXiv-v1               | arXiv-v2               |
    |  80.00% | arXiv-v2               | arXiv-v4               |
    |  80.00% | arXiv-v2               | arXiv-v3               |
    |  80.00% | arXiv-v1               | arXiv-v4               |
    |  80.00% | arXiv-v1               | arXiv-v3               |
