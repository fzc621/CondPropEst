# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import timeit
import numpy as np
import parse
import argparse
from collections import Counter
import scipy.optimize as opt
from ..lib.utils import makedirs
csv.field_size_limit(sys.maxsize)

click_field_name = ["date", "format", "paper", "ip", "mode", "uid", "session",
                    "port", "id", "useragent", "usercookies"]

query_field_name = ["date", "query", "ip", "referer", "mode", "num_results",
                    "results", "uid", "session", "port", "overlength", "id",
                    "useragent", "usercookies"]

def likelihood(p, r, c, not_c, M):
    pr = p.reshape([-1, 1]) * r
    obj = np.sum(c * np.log10(pr) + not_c * np.log10(1 - pr))
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='propensity ')
    parser.add_argument('-m', type=int, help='max pos to be estimated')
    parser.add_argument('query_path', help='query path')
    parser.add_argument('click_path', help='click path')
    parser.add_argument('output_path', help='output path')

    args = parser.parse_args()
    start = timeit.default_timer()

    M = args.m
    query_path = args.query_path
    click_path = args.click_path

    click_set = set()
    with open(click_path, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE,
                                fieldnames=click_field_name)
        for row in reader:
            uid = row['uid']
            click_set.add((uid, row['paper']))

    # query_list = []
    global_c, global_not_c = np.zeros((M, M)), np.zeros((M, M))
    with open(query_path, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE,
                                fieldnames=query_field_name)
        for row in reader:
            uid = row['uid']
            toks = row['results'].split('*')
            all_ranks = []
            for i in range(3):
                all_ranks.append(toks[3 + i].split(',')[1:])
            # c, not_c = np.zeros((M, M)), np.zeros((M, M))
            selected_ranker_id = int(toks[1])
            selected_ranker = all_ranks[selected_ranker_id]
            selected_length = len(selected_ranker)
            max_length = min(selected_length, M)
            for k in range(max_length):
                doc = selected_ranker[k]
                weight = 0
                for i in range(3):
                    ranker = all_ranks[i]
                    if len(ranker) < k:
                        continue
                    if ranker[k] == doc:
                        weight += 1

                for k_ in range(max_length):
                    if k_ == k:
                        continue

                    for i in range(3):
                        ranker = all_ranks[i]
                        if len(ranker(ranker)) < k:
                            continue
                        if ranker[k_] == doc:
                            if (uid, doc) in click_set:
                                # c[k][k_] += 1.0 / weight
                                global_c[k][k_] += 1.0 / weight
                            else:
                                # not_c[k][k_] += 1.0 / weight
                                global_not_c[k][k_] += 1.0 / weight
        # query_list.append((uid, c, not_c))

    a, b = 1e-6, 1 - 1e-6
    x0 = np.random.uniform(a, b, M * M + M)
    bnds = np.array([(a, b)] * (M * M + M))

    def f(x):
        p = x[:M]
        r = x[M:].reshape([M, M])
        r_symm = (r + r.transpose()) / 2
        return -likelihood(p, r_symm, global_c, global_not_c, M)

    ret = opt.minimize(f, x0, method='L-BFGS-B', bounds=bnds)
    prop_ = ret.x[:M]
    prop_ = prop_ / prop_[0]

    makedirs(os.path.dirname(args.output_path))
    np.savetxt(args.output_path, prop_)

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
