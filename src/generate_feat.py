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
from .lib.utils import makedirs, is_complex
csv.field_size_limit(sys.maxsize)

click_field_name = ["date", "format", "paper", "ip", "mode", "uid", "session",
                    "port", "id", "useragent", "usercookies"]

query_field_name = ["date", "query", "ip", "referer", "mode", "num_results",
                    "results", "uid", "session", "port", "overlength", "id",
                    "useragent", "usercookies"]

categories = ['eess', 'stat', 'cs', 'math', 'q-fin', 'econ', 'physics', 'q-bio', 'unknown']

article_categories = ['acc-phys', 'adap-org', 'alg-geom', 'ao-sci', 'astro-ph',
                      'bayes-an', 'chao-dyn', 'cmp-lg', 'chem-ph', 'cmp-lg',
                      'comp-gas', 'cond-mat', 'cs', 'dg-ga', 'funct-an', 'gr-qc', 'hep-ex',
                      'hep-lat', 'hep-ph', 'hep-th', 'math', 'math-ph',
                      'mtrl-th', 'nlin', 'nucl-ex', 'nucl-th', 'patt-sol',
                      'physics', 'plasm-ph', 'q-alg', 'q-bio', 'quant-ph',
                      'solv-int', 'supr-con', 'unknown']

strange_categories = ['astro-ph1', 'chao-dyn.', 'solv-int.']

len_limits = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]

session_limits = [1, 2, 5, 10, 15]

result_limits = [1, 2, 5, 10, 15, 20, 50, 100, 150, 200]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate feature')
    parser.add_argument('-m', type=int, help='max pos to be estimated')
    parser.add_argument('--complete', action='store_true', help='feat generation')
    parser.add_argument('--complex', action='store_true')
    parser.add_argument('--query_len', action='store_true')
    parser.add_argument('--session', action='store_true')
    parser.add_argument('--num_results', action='store_true')
    parser.add_argument('--result_proportion', action='store_true')
    parser.add_argument('query_path', help='query path')
    parser.add_argument('click_path', help='click path')
    parser.add_argument('feat_path', help='feat path')

    args = parser.parse_args()
    start = timeit.default_timer()

    M = args.m
    query_path = args.query_path
    click_path = args.click_path

    art_cat2idx = {}
    for article_categorie in article_categories:
        art_cat2idx[article_categorie] = len(art_cat2idx)

    for article_categorie in strange_categories:
        art_cat2idx[article_categorie] = art_cat2idx[article_categorie[:-1]]

    feats = []
    num_cat = len(categories)
    num_len_limits = len(len_limits)
    num_session_limits = len(session_limits)
    num_result_limits = len(result_limits)
    num_art_cat = len(article_categories)
    session_cnt = Counter()

    if args.complete:
        with open(query_path, 'r') as fin:
            reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE,
                                    fieldnames=query_field_name)
            for row in reader:
                query = row['query']
                feat = []
                if args.complex:
                    complex = int(is_complex(query))
                    one_hot_cat = [0] * num_cat
                    if complex:
                        cat_parser = parse.parse('(({}) AND (category:{} OR group:{}))'
                                                 , query)
                        if cat_parser:
                            cat = cat_parser[1]
                            feat.append(1)
                        else:
                            feat.append(0)
                    else:
                        cat = 'unknown'
                        feat.append(0)
                    one_hot_cat[categories.index(cat)] = 1
                    feat.extend(one_hot_cat)

                if args.query_len:
                    len_vector = [0] * num_len_limits
                    query_len = len(query.strip().split())
                    for i in range(num_len_limits):
                        if query_len <= len_limits[i]:
                            len_vector[i] = 1
                        else:
                            break
                    feat.extend(len_vector)

                if args.session:
                    session_vector = [0] * num_session_limits
                    session = row["session"]
                    session_cnt[session] += 1
                    v_session = session_cnt[session]
                    for i in range(num_session_limits):
                        if v_session <= session_limits[i]:
                            session_vector[i] = 1
                        else:
                            break
                    feat.extend(session_vector)

                if args.num_results:
                    result_vector = [0] * num_result_limits
                    num_results = int(row['num_results'])
                    for i in range(num_result_limits):
                        if num_results <= result_limits[i]:
                            result_vector[i] = 1
                        else:
                            break
                    feat.extend(result_vector)

                if args.result_proportion:
                    art_cat_vector = np.zeros(num_art_cat)
                    rankers = row['results'].split('*')[3:]
                    for ranker in rankers:
                        results = ranker.split(',')[1:]
                        for result in results:
                            if '/' in result:
                                res_cat = result.split('/')[0]
                                if '.' in res_cat:
                                    res_cat = res_cat.split('.')[0]
                                    if res_cat not in art_cat2idx:
                                        continue
                                    cat_idx = art_cat2idx[res_cat]
                                    art_cat_vector[cat_idx] += 1
                    cat_sum = sum(art_cat_vector)
                    if cat_sum > 0:
                        art_cat_vector = art_cat_vector / float(cat_sum)
                    else:
                        art_cat_vector[-1] = 1
                    feat.extend(art_cat_vector.tolist())
                feats.append(feat)
        num_queries = len(feats)
        feats = np.array(feats).reshape(num_queries,-1)
    else:
        with open(query_path, 'r') as fin:
            reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE,
                                    fieldnames=query_field_name)
            for row in reader:
                query = row['query']
                feats.append(int(is_complex(query)))
        feats = np.array(feats).reshape(-1,1)
    makedirs(os.path.dirname(args.feat_path))
    np.save(args.feat_path, feats)

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
