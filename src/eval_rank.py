# -*- coding: utf-8 -*-

import os
import sys
import timeit
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams.update({'font.size': 15})
from .lib.data_utils import load_prop
from .lib.utils import avg_rel_err, find_best_rel_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval the result')
    parser.add_argument('-k', type=int, help='#Runs')
    parser.add_argument('-m', type=int, help='#Pos')
    parser.add_argument('data_dir', help='data dir')

    args = parser.parse_args()

    start = timeit.default_timer()

    M = args.m
    columns = range(1, M + 1)

    k = args.k
    metric = {}
    for col in columns:
        metric[col] = {}

    for i in range(k):
        run_key = '#{}'.format(i)
        run_dir = os.path.join(args.data_dir, '{}'.format(i))
        gt_path = os.path.join(run_dir, 'ground_truth/set1bin.test.prop.txt')
        gt = load_prop(gt_path)
        est_dir = find_best_rel_model(os.path.join(run_dir, 'result/mlp_rel'))
        est_path = os.path.join(est_dir, 'set1bin.test.prop.txt')
        est = np.loadtxt(est_path)
        for col in columns:
            k = col - 1
            gt_k = gt[:, k]
            est_k = est[:, k]
            metric[col][run_key] = avg_rel_err(gt_k, est_k)

    metric_df = pd.DataFrame(metric, columns=columns, dtype='float64')
    metric_df.loc['avg'] = metric_df.mean()
    metric_df.loc['std'] = metric_df.std()
    metric_df.to_csv(os.path.join(args.data_dir, 'rank.csv'), float_format='%.6f')

    plt.figure()
    plt.errorbar(columns, metric_df.loc['avg'], yerr=metric_df.loc['std'], fmt='bx-', linewidth=2)
    plt.xticks(columns, columns)
    plt.xlabel('Position')
    plt.ylabel('Relative Error')
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - 0.2, x1 + 0.2, y0, y1))
    plt.savefig(os.path.join(args.data_dir, 'rank.eps'))

    end = timeit.default_timer()
    print('Runing time: {:.3f}s.'.format(end - start))
