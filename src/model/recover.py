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
from ..lib.data_utils import Query, load_log, load_feat
from ..lib.utils import makedirs
from ..lib.train_utils import train_nonlinear
from collections import defaultdict, Counter


def h(theta, k, feat):
    prod = np.dot(theta, feat)
    return 1 / pow(k, prod)

def likelihood(theta, r, X, c, not_c, M):
    exp = np.dot(X, theta).reshape(-1, 1)
    rk = np.arange(1, M + 1).reshape(1, -1)
    p = 1 / np.power(rk, exp)
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
    parser.add_argument('feat_path', help='feature path')
    parser.add_argument('log_dir', help='click log dir')
    parser.add_argument('output_dir', help='model output directory')
    args = parser.parse_args()

    start = timeit.default_timer()

    M = args.m
    D = args.d
    feat_queries = load_feat(args.feat_path)
    N = len(feat_queries)
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
                w.update({(qid, doc_id, rk):eval('n{}'.format(i))})

    c_cnt, not_c_cnt = Counter(), Counter()
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
                        c_cnt.update({(rk, rk_, qid): v})
                        not_c_cnt.update({(rk, rk_, qid): v_})


    train_queries = copy.deepcopy(feat_queries)
    c, not_c = np.zeros([N, M, M]), np.zeros([N, M, M])
    X = []
    for idx, query in enumerate(train_queries):
        qid = query._qid
        feat = query._feat
        X.append(feat)
        for k in range(M):
            for k_ in range(M):
                if k == k_:
                    continue
                c[idx][k][k_] = c_cnt[(k + 1, k_ + 1, qid)]
                not_c[idx][k][k_] = not_c_cnt[(k + 1, k_ + 1, qid)]
    X = np.array(X)

    a, b = 1e-6, 1 - 1e-6
    x0 = np.array([random.random() * (b - a) + a] * (M * M + D))
    bnds = np.array([(a, b)] * (M * M + D))

    def f(x):
        theta = x[M * M:]
        r = x[:M * M].reshape([M, M])
        r_symm = (r + r.transpose()) / 2
        return -likelihood(theta, r_symm, X, c, not_c, M)

    ret = opt.minimize(f, x0, method='L-BFGS-B', bounds=bnds)
    theta = ret.x[M * M:]

    makedirs(args.output_dir)
    model_para_path = os.path.join(args.output_dir, 'para.dat')
    with open(model_para_path, 'w') as fout:
        str_theta = list(map(lambda x: str(x), theta))
        fout.write(' '.join(str_theta))

    est_path = os.path.join(args.output_dir, 'train.est.txt')
    with open(est_path, 'w') as fout:
        for query in feat_queries:
            feat = query._feat
            qid = query._qid
            prop_ = [str(h(theta, k, feat)) for k in range(1, M + 1)]
            fout.write('qid:{} {}\n'.format(qid, ' '.join(prop_)))

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
