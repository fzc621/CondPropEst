# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import timeit
import numpy as np
import argparse
from .lib.utils import makedirs

click_field_name = ["date", "format", "paper", "ip", "mode", "uid", "session",
                    "port", "id", "useragent", "usercookies"]

query_field_name = ["date", "query", "ip", "referer", "mode", "num_results",
                    "results", "uid", "session", "port", "overlength", "id",
                    "useragent", "usercookies"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Propensity Estimation via swap intervention')
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
        reader = csv.DictReader(fin, delimiter='\t', fieldnames=click_field_name)
        for line in reader:
            if line['format'] == 'abs':
                click_set.add((line['uid'], line['paper']))

    top2k_shown = np.zeros(M)
    top2k_click = np.zeros(M)
    with open(query_path, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t', quotechar="'", fieldnames=query_field_name)
        for line in reader:
            uid = line['uid']
            num_results = int(line['num_results'])
            if num_results < M:
                continue
            results = line['results'].split('|')[-1]
            for result in results.split(','):
                toks = result.split('*')
                rk_before = int(toks[0])
                rk_after = int(toks[1])
                paper = toks[2]
                if rk_before == 0 and rk_after < M:
                    top2k_shown[rk_after] += 1
                    if (uid, paper) in click_set:
                        top2k_click[rk_after] += 1

    swap_ctr = np.zeros(M)
    prop_est = np.zeros(M)
    for i in range(M):
        swap_ctr[i] = top2k_click[i] / top2k_shown[i]

    for i in range(M):
        prop_est[i] = swap_ctr[i] / swap_ctr[0]

    makedirs(os.path.dirname(args.output_path))
    np.savetxt(args.output_path, prop_est)
    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
