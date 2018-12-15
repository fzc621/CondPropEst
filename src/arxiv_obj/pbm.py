# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import timeit
import numpy as np
import parse
import argparse
from collections import Counter
import scipy.optimize as opt
from ..lib.utils import makedirs
csv.field_size_limit(sys.maxsize)

click_field_name = ["date", "format", "paper", "ip", "mode", "uid", "session",
                    "port", "id", "useragent", "usercookies"]

query_field_name = ["date", "query", "ip", "referer", "mode", "num_results",
                    "results", "uid", "session", "port", "overlength", "id",
                    "useragent", "usercookies"]

def likelihood(p, r, c, not_c, M):
    pr = p.reshape([-1, 1]) * r
    obj = np.sum(c * np.log10(pr) + not_c * np.log10(1 - pr))
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='propensity ')
    parser.add_argument('-m', type=int, help='max pos to be estimated')
    parser.add_argument('train_path', help='train path')
    parser.add_argument('test_path', help='test path')
    parser.add_argument('output_path', help='output path')

    args = parser.parse_args()
    start = timeit.default_timer()

    M = args.m
    train_path = args.train_path
    test_path = args.test_path
    train_c, train_not_c = np.load(train_path)
    test_c, test_not_c = np.load(test_path)
    train_c = np.sum(train_c, axis=0)
    train_not_c = np.sum(train_not_c, axis=0)
    test_c = np.sum(test_c, axis=0)
    test_not_c = np.sum(test_not_c, axis=0)

    a, b = 1e-6, 1 - 1e-6
    x0 = np.random.uniform(a, b, M * M + M)
    bnds = np.array([(a, b)] * (M * M + M))

    def f(x):
        p = x[:M]
        r = x[M:].reshape([M, M])
        r_symm = (r + r.transpose()) / 2
        return -likelihood(p, r_symm, train_c, train_not_c, M)

    ret = opt.minimize(f, x0, method='L-BFGS-B', bounds=bnds)
    prop_ = ret.x[:M]
    prop_ = prop_ / prop_[0]

    opt_x = ret.x
    opt_p = opt_x[:M]
    opt_r = opt_x[M:].reshape([M, M])
    opt_r_symm = (opt_r + opt_r.transpose()) / 2
    test_loss = -likelihood(opt_p, opt_r_symm, test_c, test_not_c, M)
    makedirs(os.path.dirname(args.output_path))
    with open(args.output_path, 'w') as fout:
        fout.write('Loss: {}'.format(test_loss))

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
