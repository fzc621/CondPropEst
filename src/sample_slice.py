# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import argparse
from .lib.data_utils import Query, load_query, dump_query
from .lib.utils import makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='sample training slices from the whole dataset')
    parser.add_argument('-f', '--fraction', type=float, default=0.01,
        help='training fraction')
    parser.add_argument('train_path', help='train path')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()

    random.seed()
    start = timeit.default_timer()

    queries = load_query(args.train_path)
    random.shuffle(queries)

    train_size = len(queries)
    slice_size = int(train_size * args.fraction)
    explore_slice = sorted(queries[:slice_size], key=lambda x:x._qid)
    train_slice = sorted(queries[slice_size:], key=lambda x:x._qid)

    explore_path = os.path.join(args.output_dir, 'set1bin.exp.txt')
    train_path = os.path.join(args.output_dir, 'set1bin.train.txt')
    dump_query(explore_slice, explore_path)
    dump_query(train_slice, train_path)

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
