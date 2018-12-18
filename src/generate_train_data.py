 # -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import argparse
import numpy as np
from .lib.data_utils import Query, load_log, load_prop
from .lib.utils import *

def generate(fout, query, qid, cost, doc_id):
    fout.write('1 qid:{} cost:{} {}\n'.format(qid, cost, query._docs[doc_id][3]))
    for i in range(len(query._docs)):
        if i == doc_id:
            continue
        fout.write('0 qid:{} {}\n'.format(qid, query._docs[i][3]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate data for Prop SVM')
    parser.add_argument('-t', default=1, type=float, help='clip parameter')
    parser.add_argument('--gt', action='store_true', help='for ground truth')
    parser.add_argument('--cpbm', action='store_true', help='cpbm has to select the para')
    parser.add_argument('prop_dir', help='prop dir')
    parser.add_argument('data_path', help='data_path')
    parser.add_argument('score_path', help='score path')
    parser.add_argument('log_path', help='click log path')
    parser.add_argument('output_path', help='output data path')

    args = parser.parse_args()
    start = timeit.default_timer()
    clip_t = args.t
    queries = []
    with open(args.data_path, 'r') as fin1, open(args.score_path, 'r') as fin2:
        for line1, line2 in zip(fin1, fin2):
            line1, line2 = line1.strip(), line2.strip()
            toks = line1.split(' ', 2)
            assert len(toks) == 3
            rel = int(toks[0])
            qid = int(toks[1].split(':')[1])
            feat = toks[2]
            score = float(line2)
            if not queries or not queries[-1].equal_qid(qid):
                queries.append(Query(qid))
            doc_id = len(queries[-1]._docs)
            queries[-1].append((doc_id, score, rel, feat))

    if args.cpbm:
        prop_dir = find_best_rel_model(args.prop_dir)
    else:
        prop_dir = args.prop_dir

    os.path.join(prop_dir, 'set1bin.train.prop.txt')
    if args.gt:
        est = load_prop(prop_path)
    else:
        est = np.loadtxt(prop_path)
    M = est.shape[1]
    log = load_log(args.log_path)
    num_query = est.shape[0]
    sweep = int(len(log) / len(queries))
    qid = 0
    with open(args.output_path, 'w') as fout:
        for i in range(sweep):
            for j in range(num_query):
                log_id = i * num_query + j
                log_query = log[log_id]
                for rk, doc in enumerate(log_query._docs):
                    if doc[1] == 1:
                        if rk >= M:
                            prop = est[j][-1]
                        else:
                            prop = est[j][rk]
                        prop = max(clip_t, prop)
                        cost = 1.0 / prop
                        generate(fout, queries[j], qid, cost, doc[0])
                        qid += 1

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
