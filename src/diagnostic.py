# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from .lib.data_utils import Query, load_feat
from .lib.utils import *
from collections import Counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate the clicks')
    parser.add_argument('data_path', help='data path')
    parser.add_argument('score_path', help='score path')
    parser.add_argument('feat_path', help='feature path')

    random.seed()
    args = parser.parse_args()
    start = timeit.default_timer()
    queries = load_feat(args.feat_path)
    idx = 0
    with open(args.data_path, 'r') as fin1, open(args.score_path, 'r') as fin2:
        for line1, line2 in zip(fin1, fin2):
            line1, line2 = line1.strip(), line2.strip()
            toks = line1.split(' ', 2)
            assert len(toks) == 3
            rel = int(toks[0])
            qid = int(toks[1].split(':')[1])
            score = float(line2)
            if not queries[idx].equal_qid(qid):
                idx += 1
            assert queries[idx].equal_qid(qid)
            doc_id = len(queries[idx]._docs)
            queries[idx].append((doc_id, score, rel))
    X = np.array([q._feat for q in queries])
    Y = np.zeros((len(queries), 10))
    for i in range(len(queries)):
        query = queries[i]
        qid = query._qid
        docs = sorted(query._docs, key=lambda x: x[1], reverse=True)
        for rk, sr in enumerate(docs):
            doc_id, _, rel = sr
            if rk >= 10:
                break
            Y[i][rk] = rel
    cnt = Counter()
    for i in range(10):
        clf = LogisticRegression(multi_class='ovr', solver='lbfgs').fit(X, Y[:,i])
        abs_coef = np.abs(clf.coef_).flatten()
        imp_sort = list(reversed(np.argsort(abs_coef)))
        cnt.update(imp_sort[:15])

    print(cnt.most_common(10))
    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
