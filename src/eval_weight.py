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
from .lib.utils import read_err, find_best_rel_model, find_best_prop_model


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
    rel_metric = {}
    for col in columns:
        wo_metric[col] = {}
        mlp_metric[col] = {}
        rel_metric[col] = {}
        for i in range(k):
            run_key = '#{}'.format(i)
            run_dir = os.path.join(args.weight_dir, '{}/{}'.format(col, i))

            wo_model_path = os.path.join(run_dir,
                        'result/pbm')
            wo_err = read_err(wo_model_path, 'test')
            prop_model_path = find_best_prop_model(os.path.join(run_dir,
                        'result/mlp'))
            mlp_err = read_err(prop_model_path, 'test')
            rel_model_path = find_best_rel_model(os.path.join(run_dir,
                        'result/mlp_rel'))
            rel_err = read_err(rel_model_path, 'test')
            wo_metric[col][run_key] = wo_err
            mlp_metric[col][run_key] = mlp_err
            rel_metric[col][run_key] = rel_err

    wo_metric_df = pd.DataFrame(wo_metric, columns=columns, dtype='float64')
    wo_metric_df.loc['avg'] = wo_metric_df.mean()
    wo_metric_df.loc['std'] = wo_metric_df.std()
    wo_metric_df.to_csv(os.path.join(args.weight_dir, 'wo_result.csv'), float_format='%.6f')

    mlp_metric_df = pd.DataFrame(mlp_metric, columns=columns, dtype='float64')
    mlp_metric_df.loc['avg'] = mlp_metric_df.mean()
    mlp_metric_df.loc['std'] = mlp_metric_df.std()
    mlp_metric_df.to_csv(os.path.join(args.weight_dir, 'mlp_result.csv'), float_format='%.6f')

    rel_metric_df = pd.DataFrame(rel_metric, columns=columns, dtype='float64')
    rel_metric_df.loc['avg'] = rel_metric_df.mean()
    rel_metric_df.loc['std'] = rel_metric_df.std()
    rel_metric_df.to_csv(os.path.join(args.weight_dir, 'rel_result.csv'), float_format='%.6f')

    plt.figure()
    plt.xlabel('Strength of Context Dependence')
    plt.ylabel('Relative Error')
    plt.errorbar(columns, wo_metric_df.loc['avg'], label='PBM', yerr=wo_metric_df.loc['std'], fmt='3-.', linewidth=2)
    plt.errorbar(columns, rel_metric_df.loc['avg'], label='CPBM', color='red', yerr=rel_metric_df.loc['std'], fmt='+-', linewidth=2)
    # plt.errorbar(columns, mlp_metric_df.loc['avg'], label='CPBM w/o relevance model', color='green', yerr=mlp_metric_df.loc['std'], fmt='x--')
    plt.legend(frameon=False, loc='upper left')
    plt.xticks(columns, columns)
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - 0.2, x1 + 0.2, y0, y1))
    plt.savefig(os.path.join(args.weight_dir, 'ctx.eps'))

    end = timeit.default_timer()
    print('Runing time: {:.3f}s.'.format(end - start))
