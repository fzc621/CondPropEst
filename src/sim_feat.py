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
    parser.add_argument('-b', '--binary', type=int, default=4,
        help='#binary value')
    parser.add_argument('-f', '--float', type=int, default=6,
        help='#float value')
    parser.add_argument('train_path', help='train path')
    parser.add_argument('test_path', help='test path')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()

    random.seed()
    np.random.seed()
    start = timeit.default_timer()

    for filepath in [args.train_path, args.test_path]:
        queries = load_query(filepath)
        for query in queries:
            query._feat = []
            query._feat.extend([np.random.binomial(1, 1/(i+2))
                                    for i in range(args.binary)])
            query._feat.extend([np.random.power(i+1)
                                    for i in range(args.float)])
        filename = os.path.basename(filepath)
        featpath = os.path.join(args.output_dir,
                                    '{}.feat.txt'.format(filename[:-4]))
        dump_feat(queries, featpath)



    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
