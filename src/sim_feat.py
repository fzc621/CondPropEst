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
    parser.add_argument('input_dir', help='input dir')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()

    random.seed()
    np.random.seed()
    start = timeit.default_timer()

    filenames = [filename for filename in os.listdir(args.input_dir)
                    if filename.endswith('.txt') and not filename.endswith('feat.txt')]

    makedirs(args.output_dir)
    for filename in filenames:
        filepath = os.path.join(args.input_dir, filename)
        queries = load_query(filepath)
        for query in queries:
            query._feat = []
            query._feat.extend([np.random.binomial(1, 1/(i+2))
                                    for i in range(args.binary)])
            query._feat.extend([np.random.power(i+1)
                                    for i in range(args.float)])
        featpath = os.path.join(args.output_dir,
                                    '{}.feat.txt'.format(filename[:-4]))
        dump_feat(queries, featpath)



    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
