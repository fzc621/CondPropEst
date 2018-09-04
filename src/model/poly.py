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

l, h = 1e-6, 1 - 1e-6

def likelihood(theta, r, X, c, not_c, M, cov):
    exp = np.dot(X, theta).reshape(-1, 1)
    rk = np.arange(1, M + 1).reshape(1, -1)
    power_ = np.power(rk, exp)
    poly = np.polynomial.polynomial.polyval(X, cov)
    p = 1 / (1 + np.exp(-poly))
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
    parser.add_argument('--g', default=8, type=int,
        help='degree of polynomial')
    parser.add_argument('--test', action='store_true', help='train/test mode')
    parser.add_argument('--gt_dir', help='ground truth directory')
    parser.add_argument('npy_dir', help='numpy directory')
    parser.add_argument('model_dir', help='model directory')
    args = parser.parse_args()

    start = timeit.default_timer()


    if args.test:
        gt_path = os.path.join(args.gt_dir, 'set1bin.test.prop.txt')
        prop = load_prop(gt_path)
        M = prop.shape[1]
        model_para_path = os.path.join(args.model_dir, 'para.npy')
        theta, cov = np.load(model_para_path)
        test_feat_npy_path = os.path.join(args.npy_dir, 'test.feat.npy')
        X = np.load(test_feat_npy_path)
        exp = np.dot(X, theta)
        rk = np.arange(1, M + 1)
        poly = np.polynomial.polynomial.polyval(X, cov)
        prop_ = 1 / (1 + np.exp(-poly))
        prop_ = np.divide(prop_, prop_[:,0].reshape(-1,1))
        test_mse = _MSE(prop, prop_)
        test_prop_path = os.path.join(args.model_dir,
                                    'test.prop.mse{:.3f}.txt'.format(test_mse))
        np.savetxt(test_prop_path, prop_, fmt='%.18f')
    else:
        M = args.m
        D = args.d
        G = args.g
        click_npy_path = os.path.join(args.npy_dir, 'click.info.npy')
        c, not_c = np.load(click_npy_path)

        train_feat_npy_path = os.path.join(args.npy_dir, 'train.feat.npy')
        X = np.load(train_feat_npy_path)
        x0 = np.random.uniform(-h, h, M * M + D + G)
        bnds = np.array([(l, h)] * (M * M) + [(None, None)] * (D + G))

        def f(x):
            theta = x[M * M: M * M + D]
            cov = x[M * M + D:]
            r = x[:M * M].reshape([M, M])
            r_symm = (r + r.transpose()) / 2
            return -likelihood(theta, r_symm, X, c, not_c, M, cov)

        ret = opt.minimize(f, x0, method='L-BFGS-B', bounds=bnds)
        theta = ret.x[M * M: M * M + D]
        cov = ret.x[M * M + D:]
        print('loss: {}'.format(f(ret.x)))

        makedirs(args.model_dir)
        model_para_path = os.path.join(args.model_dir, 'para.npy')
        np.save(model_para_path, (theta, cov))

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
