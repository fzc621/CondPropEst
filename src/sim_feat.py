# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import numpy as np
import argparse
from .lib.data_utils import Query, load_query, dump_feat
from .lib.utils import makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Feature of queries simulation')
    parser.add_argument('-u', '--uniform', type=int, default=2,
        help='#uniform value')
    parser.add_argument('-n', '--normal', type=int, default=2,
        help='#normal value')
    parser.add_argument('-l', '--laplace', type=int, default=2,
        help='#laplace value')
    parser.add_argument('score0_path', help='score0 path')
    parser.add_argument('score1_path', help='score1 path')
    parser.add_argument('data_path', help='data path')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()

    random.seed()
    np.random.seed()
    start = timeit.default_timer()

    NO_REL_TAG = 100
    MAX_REL_CNT = 41 # TODO: Magic Number
    query_idx = set()
    avg_rank0, avg_rank1 = [], []
    pairs0, pairs1 = [], []
    with open(args.data_path, 'r') as fin1, open(args.score0_path, 'r') as fin2, open(args.score1_path, 'r') as fin3:
        for line1, line2, line3 in zip(fin1, fin2, fin3):
            line1, line2, line3 = line1.strip(), line2.strip(), line3.strip()
            toks = line1.split(' ', 2)
            assert len(toks) == 3
            rel = int(toks[0])
            qid = int(toks[1].split(':')[1])
            score0 = float(line2)
            score1 = float(line3)
            if qid in query_idx:
                pairs0.append((rel, score0))
                pairs1.append((rel, score1))
            else:
                r0 = sorted(pairs0, key=lambda x: x[1], reverse=True)
                r1 = sorted(pairs1, key=lambda x: x[1], reverse=True)
                cnt0, rank0 = 0, 0
                for k, p in enumerate(r0, start=1):
                    if p[0] == 1:
                        cnt0 += 1
                        rank0 += k
                if cnt0 == 0:
                    rank0 = NO_REL_TAG
                else:
                    rank0 /= cnt0
                avg_rank0.append(rank0)
                cnt1, rank1 = 0, 0
                for k, p in enumerate(r1, start=1):
                    if p[0] == 1:
                        cnt1 += 1
                        rank1 += k
                if cnt1 == 0:
                    rank1 = NO_REL_TAG
                else:
                    rank1 /= cnt1
                avg_rank1.append(rank1)
                query_idx.add(qid)
                pairs0.clear()
                pairs1.clear()
    queries = load_query(args.data_path)
    for query, a0, a1 in zip(queries, avg_rank0, avg_rank1):
        feat = []
        feat.extend(np.random.uniform(-1, 1, args.uniform))
        feat.extend(np.random.normal(size=args.normal))
        feat.extend(np.random.laplace(size=args.laplace))
        query._feat = (feat - np.mean(feat)).tolist()
        query._feat.append(float(query.get_rel_cnt()) / len(query._docs))
        query._feat.append(float(a0) / NO_REL_TAG)
        query._feat.append(float(a1) / NO_REL_TAG)
        query._feat.append(query.get_rel_cnt() / MAX_REL_CNT)
        # print(query.get_rel_cnt() / MAX_REL_CNT)
    filename = os.path.basename(args.data_path)
    featpath = os.path.join(args.output_dir,
                                '{}.feat.txt'.format(filename[:-4]))
    dump_feat(queries, featpath)



    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
