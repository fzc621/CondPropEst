# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import numpy as np
import argparse
from .lib.data_utils import Query, load_query, dump_feat, load_rel_query
from .lib.utils import makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Feature of queries simulation')
    parser.add_argument('-st', type=float, help='strength of relevance')
    parser.add_argument('-d', default=10, type=int,
        help='#dimension of feature')
    parser.add_argument('data_path', help='data path')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()

    random.seed()
    np.random.seed()
    start = timeit.default_timer()

    queries = load_rel_query(args.data_path, args.d)
    num_rel = int(args.d * args.st)
    num_rnd = args.d - num_rel
    for query in queries:
        random_feat = np.random.normal(0, 0.35, num_rnd)
        rel_feat = 2 * query._feat - 1
        if num_rel == 0:
            query._feat = random_feat.tolist()
        elif num_rnd == 0:
            query._feat = rel_feat.tolist()
        else:
            query._feat = np.concatenate((query._feat[:num_rel], random_feat)).tolist()
    filename = os.path.basename(args.data_path)
    featpath = os.path.join(args.output_dir,
                                '{}.feat.txt'.format(filename[:-4]))
    dump_feat(queries, featpath)

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
