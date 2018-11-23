# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import timeit
import numpy as np
import argparse
from .lib.utils import makedirs, is_complex

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

    query_path = args.data_path
    complex_path = os.path.join(args.output_dir, 'complex_queries_multi_swap.tsv')
    simple_path = os.path.join(args.output_dir, 'simple_queries_multi_swap.tsv')
    makedirs(os.path.dirname(complex_path))
    with open(query_path, 'r') as fin, open(complex_path, 'w') as fcomplex, open(simple_path, 'w') as fsimple:
        reader = csv.DictReader(fin, delimiter='\t', quotechar="'", fieldnames=query_field_name)
        complex_writer = csv.DictWriter(fcomplex, delimiter='\t', extrasaction='ignore',
                                        quotechar="'", fieldnames=query_field_name)
        simple_writer = csv.DictWriter(fsimple, delimiter='\t', extrasaction='ignore',
                                        quotechar="'", fieldnames=query_field_name)

        for row in reader:
            row = dict(row)
            query = row['query']
            del row[None]
            if is_complex(query):
                complex_writer.writerow(row)
            else:
                simple_writer.writerow(row)

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
