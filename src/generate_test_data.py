# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import argparse
from .lib.data_utils import Query, load_query
from .lib.utils import makedirs

def generate(fout, query, qid, doc_id):
    fout.write('1 qid:{} cost:1.0 {}\n'.format(qid, query._docs[doc_id][1]))
    for i in range(len(query._docs)):
        if i == doc_id:
            continue
        fout.write('0 qid:{} {}\n'.format(qid, query._docs[i][1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='sample training slices from the whole dataset')
    parser.add_argument('test_path', help='test path')
    parser.add_argument('output_path', help='output path')
    args = parser.parse_args()

    start = timeit.default_timer()

    queries = load_query(args.test_path)
    qid = 0
    with open(args.output_path, 'w') as fout:
        for query in queries:
            for doc_id in range(len(query._docs)):
                if query._docs[doc_id][0] == 1:
                    generate(fout, query, qid, doc_id)
                    qid += 1
    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
