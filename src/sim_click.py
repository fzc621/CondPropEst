# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import argparse
import numpy as np
from .lib.data_utils import Query, load_feat
from .lib.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate the clicks')
    parser.add_argument('-s', '--sweep', default=5, type=int,
                        help='#sweeps of the dataset')
    parser.add_argument('--epsilon_p', default=1.0, type=float,
                        help='the prob of users click on a relevant result')
    parser.add_argument('--epsilon_n', default=0.1, type=float,
                        help='the prob of users click on a irrelevant result')
    parser.add_argument('para_path', help='func parameter path')
    parser.add_argument('data_path', help='data path')
    parser.add_argument('score_path', help='score path')
    parser.add_argument('feat_path', help='feature path')
    parser.add_argument('log_path', help='click log path')

    random.seed()
    args = parser.parse_args()
    start = timeit.default_timer()

    ep = args.epsilon_p
    en = args.epsilon_n
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

    w = read_para(args.para_path)
    makedirs(os.path.dirname(args.log_path))
    with open(args.log_path, 'w') as fout:
        for i in range(args.sweep):
            for query in queries:
                qid = query._qid
                feat = query._feat
                docs = sorted(query._docs, key=lambda x: x[1], reverse=True)
                for rk, sr in enumerate(docs, start=1):
                    doc_id, _, rel = sr
                    pr = cal_prob(w, feat, rk)
                    if i == 0 and rk == 3:
                        print(pr)
                    clicked = False
                    if prob_test(pr):
                        if rel:
                            if prob_test(ep):
                                clicked = True
                        else:
                            if prob_test(en):
                                clicked = True
                    if clicked:
                        fout.write('{} qid:{} {}\n'.format(1, qid, doc_id))
                    else:
                        fout.write('{} qid:{} {}\n'.format(0, qid, doc_id))

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
