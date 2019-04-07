# -*- coding: utf-8 -*-

import os
import sys
import timeit
import parse
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams.update({'font.size': 15})
def read_avg_rank(path):
    with open(path) as fin:
        line = fin.readline().rstrip()
        rank = parse.parse('Avg Rank of Positive Examples: {} (via SNIPS Estimator [Swaminathan & Joachims, 2015d])', line)
        return float(rank[0])

def get_model_avg_rank(run_dir, model):
    valid_logs = [r for r in os.scandir(run_dir)
                if r.name.startswith('valid_{}'.format(model))]
    min_rank = 100
    min_valid_log = []
    for valid_log in valid_logs:
        rank = read_avg_rank(valid_log)
        if min_rank > rank:
            min_rank = rank
            min_valid_log = [valid_log.name]
        elif min_rank == rank:
            min_valid_log.append(valid_log.name)

    min_test_log = ['test_{}'.format(log[6:]) for log in min_valid_log]

    min_test_rank = 100
    for test_log in min_test_log:
        log_path = os.path.join(run_dir, test_log)
        rank = read_avg_rank(log_path)
        if rank < min_test_rank:
            min_test_rank = rank
    return min_test_rank

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval the result')
    parser.add_argument('-k', type=int, help='#Runs')
    parser.add_argument('learn_dir', help='learning data dir')

    args = parser.parse_args()
    start = timeit.default_timer()

    params = [c for c in os.scandir(args.learn_dir)
                if not c.name.startswith('.') and c.is_dir()]
    columns = sorted([c.name for c in params], key=float)
    k = args.k
    pbm_metric = {}
    cpbm_metric = {}
    for col in columns:
        pbm_metric[col] = {}
        cpbm_metric[col] = {}
        for i in range(k):
            run_key = '#{}'.format(i)
            run_dir = os.path.join(args.learn_dir, '{}/{}/learn'.format(col, i))

            click_cnt_path = os.path.join(run_dir, 'log0.cnt')
            with open(click_cnt_path) as fin:
                click_cnt = int(fin.readline())

            pbm_rk = get_model_avg_rank(run_dir, 'pbm')
            cpbm_rk = get_model_avg_rank(run_dir, 'cpbm')
            gt_rk = get_model_avg_rank(run_dir, 'gt')

            pbm_metric[col][run_key] = pbm_rk - gt_rk
            cpbm_metric[col][run_key] = cpbm_rk - gt_rk

    pbm_metric_df = pd.DataFrame(pbm_metric, columns=columns, dtype='float64')
    pbm_metric_df.loc['avg'] = pbm_metric_df.mean()
    pbm_metric_df.loc['std'] = pbm_metric_df.std()
    pbm_metric_df.to_csv(os.path.join(args.learn_dir, 'pbm_result.csv'), float_format='%.4f')

    cpbm_metric_df = pd.DataFrame(cpbm_metric, columns=columns, dtype='float64')
    cpbm_metric_df.loc['avg'] = cpbm_metric_df.mean()
    cpbm_metric_df.loc['std'] = cpbm_metric_df.std()
    cpbm_metric_df.to_csv(os.path.join(args.learn_dir, 'cpbm_result.csv'), float_format='%.4f')

    N = len(columns)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25       # the width of the bars
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab20")
    pbm_rects = ax.bar(ind, pbm_metric_df.loc['avg'], width, color=cmap.colors[1], yerr=pbm_metric_df.loc['std'], linewidth=0)
    cpbm_rects = ax.bar(ind+width, cpbm_metric_df.loc['avg'], width, color=cmap.colors[6], yerr=cpbm_metric_df.loc['std'], linewidth=0)

    ax.set_xlabel('#Training queries')
    ax.set_ylabel('Difference in AvgRank')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('11473','22946','57365'))
    ax.axhline(0, color='black')

    ax.legend((pbm_rects[0], cpbm_rects[0]), ('PBM', 'CPBM'), frameon=False, markerfirst=False)
    plt.savefig(os.path.join(args.learn_dir, 'learn.eps'))

    end = timeit.default_timer()
    print('Runing time: {:.3f}s.'.format(end - start))
