# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import timeit
import numpy as np
import argparse
import multiprocessing as mp
import scipy.optimize as opt
from ..lib.utils import makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Propensity Estimation via swap intervention')
    parser.add_argument('-m', type=int, help='max pos to be estimated')
    parser.add_argument('-p', type=float, default=0.95, help='confdence probability')
    parser.add_argument('-n', type=int, default=1000, help='num of bootstrap samples')
    parser.add_argument('result_dir', help='result dir')
    args = parser.parse_args()
    start = timeit.default_timer()

    M = args.m
    n_bootstrap = args.n

    lo = int(n_bootstrap * ((1 - args.p) / 2))
    mi = int(n_bootstrap * 0.5)
    hi = n_bootstrap - lo

    simple_prop_list = []
    for i in range(n_bootstrap):
        simple_prop_path = os.path.join(args.result_dir,
                                        '{}/simple_bootstrap.txt'.format(i))
        simple_prop = np.loadtxt(simple_prop_path)
        simple_prop_list.append(simple_prop)

    complex_prop_list = []
    for i in range(n_bootstrap):
        complex_prop_path = os.path.join(args.result_dir,
                                        '{}/complex_bootstrap.txt'.format(i))
        complex_prop = np.loadtxt(complex_prop_path)
        complex_prop_list.append(complex_prop)

    simple_perc_conf = np.zeros((M, 3))
    for i in range(M):
        p = []
        for prop in simple_prop_list:
            p.append(prop[i])
        p.sort()
        simple_perc_conf[i][0] = p[lo]
        simple_perc_conf[i][1] = p[mi]
        simple_perc_conf[i][2] = p[hi]

    simple_merge_path = os.path.join(args.result_dir, 'simple_bootstrap.txt')
    np.savetxt(simple_merge_path, simple_perc_conf)

    complex_perc_conf = np.zeros((M, 3))
    for i in range(M):
        p = []
        for prop in complex_prop_list:
            p.append(prop[i])
        p.sort()
        complex_perc_conf[i][0] = p[lo]
        complex_perc_conf[i][1] = p[mi]
        complex_perc_conf[i][2] = p[hi]
    complex_merge_path = os.path.join(args.result_dir, 'complex_bootstrap.txt')
    np.savetxt(complex_merge_path, complex_perc_conf)

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
