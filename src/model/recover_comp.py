# -*- coding: utf-8 -*-

import os
import sys
import math
import copy
import random
import timeit
import argparse
import numpy as np
import scipy.optimize as opt
from ..lib.data_utils import Query, load_feat, load_prop
from ..lib.utils import makedirs, _MSE
from collections import defaultdict, Counter

def likelihood(theta, r, X, c, not_c, M):
    a = np.dot(X, theta).reshape(-1, 1)
    rk = np.arange(M).reshape(1, -1)
    p = 1 / (a * rk + 1)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pr = p.reshape([-1, M, 1]) * r
    obj = np.sum(c * np.log10(pr) + not_c * np.log10(1 - pr))
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='propensity estimation w/o condition')
    parser.add_argument('-m', default=10, type=int,
        help='number of top positions for which estimates are desired')
    parser.add_argument('-d', default=10, type=int,
        help='dimension of feature')
    parser.add_argument('--test', action='store_true', help='train/test mode')
    parser.add_argument('--gt_dir', help='ground truth directory')
    parser.add_argument('npy_dir', help='numpy directory')
    parser.add_argument('model_dir', help='model directory')
    args = parser.parse_args()

    start = timeit.default_timer()
    if args.test:
        gt_path = os.path.join(args.gt_dir, 'set1bin.test.prop.txt')
        p = load_prop(gt_path)
        M = p.shape[1]
        model_para_path = os.path.join(args.model_dir, 'para.npy')
        theta = np.load(model_para_path)
        test_feat_npy_path = os.path.join(args.npy_dir, 'test.feat.npy')
        X = np.load(test_feat_npy_path)
        a = np.dot(X, theta).reshape(-1, 1)
        rk = np.arange(M).reshape(1, -1)
        p_ = 1 / (a * rk + 1)
        p_ = p_ / p_[:,0].reshape(-1,1)

        test_mse = _MSE(p, p_)
        test_prop_path = os.path.join(args.model_dir,
                                    'test.prop.mse{:.3f}.txt'.format(test_mse))
        np.savetxt(test_prop_path, p_, fmt='%.18f')
    else:
        M = args.m
        D = args.d
        click_npy_path = os.path.join(args.npy_dir, 'click.info.npy')
        c, not_c = np.load(click_npy_path)

        train_feat_npy_path = os.path.join(args.npy_dir, 'train.feat.npy')
        X = np.load(train_feat_npy_path)
        a, b = 1e-6, 1 - 1e-6
        x0 = np.array([random.random() * (b - a) + a for i in range(M * M + D)])
        bnds = np.array([(a, b)] * (M * M) + [(None, None)] * D)

        def f(x):
            theta = x[M * M:]
            r = x[:M * M].reshape([M, M])
            r_symm = (r + r.transpose()) / 2
            return -likelihood(theta, r_symm, X, c, not_c, M)

        ret = opt.minimize(f, x0, method='L-BFGS-B', bounds=bnds)
        theta = ret.x[M * M:]
        print('loss: {}'.format(f(ret.x)))

        makedirs(args.model_dir)
        model_para_path = os.path.join(args.model_dir, 'para.npy')
        np.save(model_para_path, theta)

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
