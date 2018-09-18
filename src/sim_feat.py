# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import numpy as np
import argparse
from .lib.data_utils import Query, load_query, dump_feat
from .lib.utils import makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Feature of queries simulation')
    parser.add_argument('-u', '--uniform', type=int, default=3,
        help='#uniform value')
    parser.add_argument('-n', '--normal', type=int, default=3,
        help='#normal value')
    parser.add_argument('-l', '--laplace', type=int, default=4,
        help='#laplace value')
    parser.add_argument('data_dir', help='data dir')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()

    random.seed()
    np.random.seed()
    start = timeit.default_timer()

    files = [os.path.join(args.data_dir, filepath)
                    for filepath in os.listdir(args.data_dir)
                        if not filepath.endswith('feat.txt')
                            and filepath.endswith('.txt')]

    for filepath in files:
        queries = load_query(filepath)
        for query in queries:
            feat = []
            feat.extend(np.random.uniform(-1, 1, args.uniform))
            feat.extend(np.random.normal(size=args.normal))
            feat.extend(np.random.laplace(size=args.laplace))
            query._feat = (feat - np.mean(feat)).tolist()
        filename = os.path.basename(filepath)
        featpath = os.path.join(args.output_dir,
                                    '{}.feat.txt'.format(filename[:-4]))
        dump_feat(queries, featpath)



    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
