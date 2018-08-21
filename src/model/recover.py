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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='propensity estimation w/o condition')
    parser.add_argument('-n', default=10, type=int,
        help='number of top positions for which estimates are desired')
    parser.add_argument('-d', default=10, type=int,
        help='dimension of feature')
    parser.add_argument('-s', default=256, type=int,
        help='batch size')
    parser.add_argument('feat_path', help='feature path')
    parser.add_argument('log_dir', help='click log dir')
    parser.add_argument('output_dir', help='model output directory')
    args = parser.parse_args()

    start = timeit.default_timer()

    M = args.n
    D = args.d
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

    c = Counter()
    not_c = Counter()
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
                        c.update({(rk, rk_, qid): v})
                        not_c.update({(rk, rk_, qid): v_})

    feat_queries = load_feat(args.feat_path)
    theta_ = train_nonlinear(M, c, not_c, feat_queries, args.s, D, h)

    makedirs(args.output_dir)
    model_para_path = os.path.join(args.output_dir, 'para.dat')
    with open(model_para_path, 'w') as fout:
        str_theta = list(map(lambda x: str(x), theta_))
        fout.write(' '.join(str_theta))

    est_path = os.path.join(args.output_dir, 'est.txt')
    with open(est_path, 'w') as fout:
        for query in feat_queries:
            feat = query._feat
            qid = query._qid
            prop_ = [str(h(theta_, k, feat)) for k in range(1, M + 1)]
            fout.write('qid:{} {}\n'.format(qid, ' '.join(prop_)))

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
