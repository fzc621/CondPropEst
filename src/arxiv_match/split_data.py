# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import timeit
import numpy as np
import argparse
from ..lib.utils import makedirs, is_complex
csv.field_size_limit(sys.maxsize)

query_field_name = ["date", "query", "ip", "referer", "mode", "num_results",
                    "results", "uid", "session", "port", "overlength", "id",
                    "useragent", "usercookies"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split complex queries & simple queries')
    parser.add_argument('data_path', help='data path')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()
    start = timeit.default_timer()

    random.seed()
    query_path = args.data_path
    query_name = os.path.basename(args.data_path)
    train_path = os.path.join(args.output_dir, 'train_{}'.format(query_name))
    valid_path = os.path.join(args.output_dir, 'valid_{}'.format(query_name))
    test_path = os.path.join(args.output_dir, 'test_{}'.format(query_name))
    makedirs(args.output_dir)
    with open(query_path, 'r') as fin, open(train_path, 'w') as ftrain, open(valid_path, 'w') as fvalid, open(test_path, 'w') as ftest:
        reader = csv.DictReader(fin, delimiter='\t', quotechar="'", fieldnames=query_field_name)
        train_writer = csv.DictWriter(ftrain, delimiter='\t', extrasaction='ignore',
                                        quotechar="'", fieldnames=query_field_name)
        valid_writer = csv.DictWriter(fvalid, delimiter='\t', extrasaction='ignore',
                                        quotechar="'", fieldnames=query_field_name)
        test_writer = csv.DictWriter(ftest, delimiter='\t', extrasaction='ignore',
                                        quotechar="'", fieldnames=query_field_name)
        for row in reader:
            row = dict(row)
            query = row['query']
            del row[None]
            v = random.random()
            if v < 0.8:
                train_writer.writerow(row)
            elif v < 0.9:
                valid_writer.writerow(row)
            else:
                test_writer.writerow(row)
    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
