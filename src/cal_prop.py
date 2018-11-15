# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import argparse
import numpy as np
from .lib.data_utils import Query, load_feat
from .lib.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate query propensity')
    parser.add_argument('-n', default=10, type=int,
        help='#top positions for which estimates are desired')
    parser.add_argument('-d', default=699, type=int,
        help='#dimension of feature')
    parser.add_argument('-m', choices=['exp', 'power', 'comp'],
        help='simulation function')
    parser.add_argument('-w', type=float,
        help='weight range')
    parser.add_argument('para_path', help='func parameter path')
    parser.add_argument('feat_path', help='feature path')
    parser.add_argument('prop_path', help='query propensity path')

    random.seed()
    args = parser.parse_args()
    start = timeit.default_timer()

    queries = load_feat(args.feat_path)
    M = args.n
    D = args.d
    r = args.w
    w = read_para(args.para_path, D, r)
    makedirs(os.path.dirname(args.prop_path))
    with open(args.prop_path, 'w') as fout:
        for query in queries:
            qid = query._qid
            feat = query._feat
            prop = [cal_prob(w, feat, k, args.m) for k in range(1, M + 1)]
            # str_prop = [str(p / prop[0]) for p in prop]
            str_prop = [str(p) for p in prop]
            fout.write('qid:{} {}\n'.format(qid, ' '.join(str_prop)))

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
