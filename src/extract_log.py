# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import timeit
import numpy as np
import argparse
from .lib.utils import makedirs, is_complex
csv.field_size_limit(sys.maxsize)

click_field_name = ["date", "format", "paper", "ip", "mode", "uid", "session",
                    "port", "id", "useragent", "usercookies"]

query_field_name = ["date", "query", "ip", "referer", "mode", "num_results",
                    "results", "uid", "session", "port", "overlength", "id",
                    "useragent", "usercookies"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate log and feature')
    parser.add_argument('-m', type=int, help='max pos to be estimated')
    parser.add_argument('query_path', help='query path')
    parser.add_argument('click_path', help='click path')
    parser.add_argument('feat_path', help='feat path')
    parser.add_argument('info_path')

    args = parser.parse_args()
    start = timeit.default_timer()

    M = args.m
    query_path = args.query_path
    click_path = args.click_path
    uid2qid = {}
    click_set = set()

    with open(click_path, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE,
                                fieldnames=click_field_name)
        for row in reader:
            if row['format'] == 'abs':
                uid = row['uid']
                click_set.add((uid, row['paper']))

    n = np.zeros(3)
    feats = []
    # feat generation
    with open(query_path, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE,
                                fieldnames=query_field_name)
        for row in reader:
            query = row['query']
            feats.append(int(is_complex(query)))
            toks = row['results'].split('*')
            selected_ranker_id = int(toks[1])
            n[selected_ranker_id] += 1
    num_queries = len(feats)
    feats = np.array(feats).reshape(-1,1)
    np.save(args.feat_path, feats)

    c, not_c = np.zeros((num_queries, M, M)), np.zeros((num_queries, M, M))
    cnt = 0
    with open(query_path, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE,
                                fieldnames=query_field_name)
        for row in reader:
            uid = row['uid']
            toks = row['results'].split('*')
            all_ranks = []
            for i in range(3):
                all_ranks.append(toks[3 + i].split(',')[1:])
            selected_ranker_id = int(toks[1])
            selected_ranker = all_ranks[selected_ranker_id]
            selected_length = len(selected_ranker)
            max_length = min(selected_length, M)
            for k in range(max_length):
                doc = selected_ranker[k]
                weight = 0
                for i in range(3):
                    ranker = all_ranks[i]
                    if ranker[k] == doc:
                        weight += n[i]

                for k_ in range(max_length):
                    if k_ == k:
                        continue

                    for i in range(3):
                        ranker = all_ranks[i]
                        if ranker[k_] == doc:
                            if (uid, doc) in click_set:
                                c[cnt][k][k_] = 1.0 / weight
                            else:
                                not_c[cnt][k][k_] = 1.0 / weight
            cnt += 1

    assert cnt == num_queries
    np.save(args.info_path, (c, not_c))
    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
