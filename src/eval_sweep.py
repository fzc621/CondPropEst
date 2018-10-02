# -*- coding: utf-8 -*-

import os
import sys
import timeit
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from .lib.utils import read_test_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval the result')
    parser.add_argument('-k', type=int, help='#Runs')
    parser.add_argument('sweep_dir', help='sweep data dir')

    args = parser.parse_args()

    start = timeit.default_timer()

    params = [c for c in os.scandir(args.sweep_dir)
                if not c.name.startswith('.') and c.is_dir()]
    columns = sorted([c.name for c in params], key=float)
    float_columns = list(map(float, columns))
    # columns = sorted([c.name for c in params], key=float)
    k = args.k
    wo_metric = {}
    mlp_metric = {}
    rel_metric = {}
    for col in columns:
        wo_metric[col] = {}
        mlp_metric[col] = {}
        rel_metric[col] = {}
        for i in range(k):
            run_key = '#{}'.format(i)
            run_dir = os.path.join(args.sweep_dir, '{}/{}'.format(col, i))

            wo_err_path = os.path.join(run_dir,
                        'result/wo_cond/test.txt')
            wo_err = read_test_err(wo_err_path)
            mlp_err_path = os.path.join(run_dir,
                        'result/ann/mlp_power_best/test.txt')
            mlp_err = read_test_err(mlp_err_path)
            rel_err_path = os.path.join(run_dir,
                        'result/ann/mlp_best_rel/test.txt')
            rel_err = read_test_err(rel_err_path)
            wo_metric[col][run_key] = wo_err
            mlp_metric[col][run_key] = mlp_err
            rel_metric[col][run_key] = rel_err

    wo_metric_df = pd.DataFrame(wo_metric, columns=columns, dtype='float64')
    wo_metric_df.loc['avg'] = wo_metric_df.mean()
    wo_metric_df.loc['std'] = wo_metric_df.std()
    wo_metric_df.to_csv(os.path.join(args.sweep_dir, 'wo_result.csv'), float_format='%.6f')

    mlp_metric_df = pd.DataFrame(mlp_metric, columns=columns, dtype='float64')
    mlp_metric_df.loc['avg'] = mlp_metric_df.mean()
    mlp_metric_df.loc['std'] = mlp_metric_df.std()
    mlp_metric_df.to_csv(os.path.join(args.sweep_dir, 'mlp_result.csv'), float_format='%.6f')

    rel_metric_df = pd.DataFrame(rel_metric, columns=columns, dtype='float64')
    rel_metric_df.loc['avg'] = rel_metric_df.mean()
    rel_metric_df.loc['std'] = rel_metric_df.std()
    rel_metric_df.to_csv(os.path.join(args.sweep_dir, 'rel_result.csv'), float_format='%.6f')

    plt.figure()
    plt.errorbar(columns, rel_metric_df.loc['avg'], label='w/ relevance', yerr=rel_metric_df.loc['std'])
    plt.xticks(columns, columns)
    plt.xlabel('Amount of Logged Data')
    plt.ylabel('Relative Error')
    plt.savefig(os.path.join(args.sweep_dir, 'rel.pdf'))
    plt.errorbar(columns, wo_metric_df.loc['avg'], label='w/o features', yerr=wo_metric_df.loc['std'])
    plt.errorbar(columns, mlp_metric_df.loc['avg'], label='w/o relevance', yerr=mlp_metric_df.loc['std'])
    plt.legend()
    plt.savefig(os.path.join(args.sweep_dir, 'sweep.pdf'))

    end = timeit.default_timer()
    print('Runing time: {:.3f}s.'.format(end - start))
