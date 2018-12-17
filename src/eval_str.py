# -*- coding: utf-8 -*-

import os
import sys
import timeit
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.style.use('classic')
from .lib.utils import read_err, find_best_rel_model, find_best_prop_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval the result')
    parser.add_argument('-k', type=int, help='#Runs')
    parser.add_argument('str_dir', help='Strength data dir')

    args = parser.parse_args()

    start = timeit.default_timer()

    params = [c for c in os.scandir('{}/1/strength'.format(args.str_dir))
                if not c.name.startswith('.') and c.is_dir()]
    columns = sorted([c.name for c in params], key=float)

    pbm_metric = {}
    cpbm_wo_rel_metric = {}
    cpbm_metric = {}
    imp_metric = {}

    for col in columns:
        pbm_metric[col] = {}
        cpbm_wo_rel_metric[col] = {}
        cpbm_metric[col] = {}
        imp_metric[col] = {}
        for i in range(args.k):
            run_key = '#{}'.format(i)
            run_dir = os.path.join(args.str_dir, '{}/strength'.format(i))

            pbm_model_path = os.path.join(run_dir,
                        '{}/result/pbm'.format(col))
            pbm_err = read_err(pbm_model_path, 'test')
            cpbm_wo_rel_model_path = find_best_prop_model(os.path.join(run_dir,
                        '{}/result/mlp'.format(col)))
            cpbm_wo_rel_err = read_err(cpbm_wo_rel_model_path, 'test')
            cpbm_model_path = find_best_rel_model(os.path.join(run_dir,
                        '{}/result/mlp_rel'.format(col)))
            cpbm_err = read_err(cpbm_model_path, 'test')
            pbm_metric[col][run_key] = pbm_err
            cpbm_wo_rel_metric[col][run_key] = cpbm_wo_rel_err
            cpbm_metric[col][run_key] = cpbm_err
            imp_metric[col][run_key] = cpbm_wo_rel_err - cpbm_err


    pbm_metric_df = pd.DataFrame(pbm_metric, columns=columns, dtype='float64')
    pbm_metric_df.loc['avg'] = pbm_metric_df.mean()
    pbm_metric_df.loc['std'] = pbm_metric_df.std()
    pbm_metric_df.to_csv(os.path.join(args.str_dir, 'pbm_result.csv'), float_format='%.6f')

    cpbm_wo_rel_metric_df = pd.DataFrame(cpbm_wo_rel_metric, columns=columns, dtype='float64')
    cpbm_wo_rel_metric_df.loc['avg'] = cpbm_wo_rel_metric_df.mean()
    cpbm_wo_rel_metric_df.loc['std'] = cpbm_wo_rel_metric_df.std()
    cpbm_wo_rel_metric_df.to_csv(os.path.join(args.str_dir, 'cpbm_wo_rel_result.csv'), float_format='%.6f')

    cpbm_metric_df = pd.DataFrame(cpbm_metric, columns=columns, dtype='float64')
    cpbm_metric_df.loc['avg'] = cpbm_metric_df.mean()
    cpbm_metric_df.loc['std'] = cpbm_metric_df.std()
    cpbm_metric_df.to_csv(os.path.join(args.str_dir, 'cpbm_result.csv'), float_format='%.6f')

    imp_metric_df = pd.DataFrame(imp_metric, columns=columns, dtype='float64')
    imp_metric_df.loc['avg'] = imp_metric_df.mean()
    imp_metric_df.loc['std'] = imp_metric_df.std()
    imp_metric_df.to_csv(os.path.join(args.str_dir, 'imp_result.csv'), float_format='%.6f')

    plt.figure()
    plt.xlabel('Strength of Relevance Dependence')
    plt.ylabel('Error Reduction')
    # plt.errorbar(columns, pbm_metric_df.loc['avg'], label='PBM', yerr=pbm_metric_df.loc['std'], fmt='3-.')
    # plt.errorbar(columns, cpbm_metric_df.loc['avg'], label='CPBM', color='red', yerr=cpbm_metric_df.loc['std'], fmt='+-')
    # plt.errorbar(columns, mlp_metric_df.loc['avg'], label='CPBM w/o relevance model', color='green', yerr=mlp_metric_df.loc['std'], fmt='x--')
    plt.errorbar(columns, imp_metric_df.loc['avg'], label='Improvement')
    plt.legend(frameon=False, loc='upper left')
    plt.xticks(columns, columns)
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - 0.2, x1 + 0.2, y0, y1))
    plt.savefig(os.path.join(args.str_dir, 'strength.eps'))

    end = timeit.default_timer()
    print('Runing time: {:.3f}s.'.format(end - start))
