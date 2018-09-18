# -*- coding: utf-8 -*-

import os
import sys
import math
import random
import timeit
import argparse
import numpy as np
import scipy.optimize as opt
from ..lib.data_utils import Query, load_log, load_prop
from ..lib.utils import makedirs, avg_rel_err
from collections import defaultdict, Counter

def likelihood(p, r, c, not_c, M):
    pr = p.reshape([-1, 1]) * r
    obj = np.sum(c * np.log10(pr) + not_c * np.log10(1 - pr))
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='propensity estimation w/o condition')
    parser.add_argument('-n', default=10, type=int,
                        help='number of top positions for which estimates are desired')
    parser.add_argument('--test', action='store_true', help='train/test mode')
    parser.add_argument('--gt_dir', help='ground truth directory')
    parser.add_argument('--log_dir', help='click log dir')
    parser.add_argument('model_dir', help='model directory')
    args = parser.parse_args()

    start = timeit.default_timer()

    if args.test:
        gt_path = os.path.join(args.gt_dir, 'set1bin.test.prop.txt')
        prop = load_prop(gt_path)
        model_para_path = os.path.join(args.model_dir, 'para.npy')
        para_ = np.load(model_para_path)
        N = prop.shape[0]
        prop_ = np.tile(para_, (N, 1))

        print('Relative Error on test set: {}'.format(avg_rel_err(prop, prop_)))
        test_prop_path = os.path.join(args.model_dir, 'set1.test.prop.txt')

        np.savetxt(test_prop_path, prop_, fmt='%.18f')
    else:
        M = args.n
        log0_path = os.path.join(args.log_dir, 'log0.txt')
        log1_path = os.path.join(args.log_dir, 'log1.txt')
        log0 = load_log(log0_path)
        log1 = load_log(log1_path)

        S = defaultdict(set)
        for q0, q1 in zip(log0, log1):
            assert q0._qid == q1._qid
            qid = q0._qid
            docs0 = q0._docs
            docs1 = q1._docs
            for rk0, doc0 in enumerate(docs0, start=1):
                if rk0 > M:
                    break
                doc_id0, _ = doc0
                for rk1, doc1 in enumerate(docs1, start=1):
                    if rk1 > M:
                        break
                    if rk1 == rk0:
                        continue
                    doc_id1, _ = doc1
                    if doc_id1 == doc_id0:
                        S[(rk0, rk1)].add((qid, doc_id0))
                        S[(rk1, rk0)].add((qid, doc_id0))
                        break

        n0 = len(log0)
        n1 = len(log1)
        assert n0 == n1
        w = Counter()
        for i in range(2):
            logger = eval('log{}'.format(i))
            for q in logger:
                qid = q._qid
                docs = q._docs
                for rk, doc in enumerate(docs, start=1):
                    if rk > M:
                        break
                    doc_id, _ = doc
                    w.update({(qid, doc_id, rk): eval('n{}'.format(i))})

        c, not_c = np.zeros([M, M]), np.zeros([M, M])
        for i in range(2):
            logger = eval('log{}'.format(i))
            for q in logger:
                qid = q._qid
                docs = q._docs
                for rk, doc in enumerate(docs, start=1):
                    if rk > M:
                        break
                    doc_id, delta = doc
                    v = delta / w[(qid, doc_id, rk)]
                    v_ = (1 - delta) / w[(qid, doc_id, rk)]
                    for rk_ in range(1, M + 1):
                        if (qid, doc_id) in S[(rk, rk_)]:
                            c[rk - 1][rk_ - 1] += v
                            not_c[rk - 1][rk_ - 1] += v_

        a, b = 1e-6, 1 - 1e-6
        x0 = np.random.uniform(a, b, M * M + M)
        bnds = np.array([(a, b)] * (M * M + M))

        def f(x):
            p = x[:M]
            r = x[M:].reshape([M, M])
            r_symm = (r + r.transpose()) / 2
            return -likelihood(p, r_symm, c, not_c, M)

        ret = opt.minimize(f, x0, method='L-BFGS-B', bounds=bnds)
        prop_ = ret.x[:M]
        prop_ = prop_ / prop_[0]

        makedirs(args.model_dir)
        model_para_path = os.path.join(args.model_dir, 'para.npy')
        np.save(model_para_path, prop_)

        gt_path = os.path.join(args.gt_dir, 'set1bin.train.prop.txt')
        prop = load_prop(gt_path)
        print('Relative Error on training set: {}'.format(avg_rel_err(prop, prop_)))

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
