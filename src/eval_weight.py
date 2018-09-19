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
    parser.add_argument('weight_dir', help='weight data dir')

    args = parser.parse_args()

    start = timeit.default_timer()

    params = [c for c in os.scandir(args.weight_dir)
                if not c.name.startswith('.') and c.is_dir()]
    columns = sorted([c.name for c in params], key=float)

    k = args.k
    wo_metric = {}
    mlp_metric = {}
    for col in columns:
        wo_metric[col] = {}
        mlp_metric[col] = {}
        for i in range(k):
            run_key = '#{}'.format(i)
            run_dir = os.path.join(args.weight_dir, '{}/{}'.format(col, i))

            wo_err_path = os.path.join(run_dir,
                        'result/wo_cond/test.txt')
            wo_err = read_test_err(wo_err_path)
            mlp_err_path = os.path.join(run_dir,
                        'result/ann/mlp_power_best/test.txt')
            mlp_err = read_test_err(mlp_err_path)
            wo_metric[col][run_key] = wo_err
            mlp_metric[col][run_key] = mlp_err

    wo_metric_df = pd.DataFrame(wo_metric, columns=columns, dtype='float64')
    wo_metric_df.loc['avg'] = wo_metric_df.mean()
    wo_metric_df.loc['std'] = wo_metric_df.std()
    wo_metric_df.to_csv(os.path.join(args.weight_dir, 'wo_result.csv'), float_format='%.6f')

    mlp_metric_df = pd.DataFrame(mlp_metric, columns=columns, dtype='float64')
    mlp_metric_df.loc['avg'] = mlp_metric_df.mean()
    mlp_metric_df.loc['std'] = mlp_metric_df.std()
    mlp_metric_df.to_csv(os.path.join(args.weight_dir, 'mlp_result.csv'), float_format='%.6f')

    plt.figure()
    plt.errorbar(columns, wo_metric_df.loc['avg'], label='w/o features', yerr=wo_metric_df.loc['std'])
    plt.errorbar(columns, mlp_metric_df.loc['avg'], label='w/ features', ls='-.', yerr=mlp_metric_df.loc['std'])
    plt.xticks(columns, columns)
    plt.legend()
    plt.xlabel('Half Range of ||W||')
    plt.ylabel('Relative Error')
    plt.savefig(os.path.join(args.weight_dir, 'weight.pdf'))

    end = timeit.default_timer()
    print('Runing time: {:.3f}s.'.format(end - start))
