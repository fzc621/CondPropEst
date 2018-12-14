# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import timeit
import numpy as np
import argparse
import multiprocessing as mp
from .lib.utils import makedirs

click_field_name = ["date", "format", "paper", "ip", "mode", "uid", "session",
                    "port", "id", "useragent", "usercookies"]

query_field_name = ["date", "query", "ip", "referer", "mode", "num_results",
                    "results", "uid", "session", "port", "overlength", "id",
                    "useragent", "usercookies"]

class EstWorker(mp.Process):

    def __init__(self, task_queue, M, click_set, res_list):
        super(EstWorker, self).__init__()
        self._task_queue = task_queue
        self._M = M
        self._click_set = click_set
        self._res_list = res_list

    def run(self):
        task_queue = self._task_queue
        click_set = self._click_set
        res_list = self._res_list
        M = self._M
        name = self.name

        cnt = 0
        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                print('{}: Processed {} tasks'.format(name, cnt))
                break
            query_set = task
            top2k_shown = np.zeros(M)
            top2k_click = np.zeros(M)
            for uid, results in query_set:
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
            res_list.append(prop_est)
            task_queue.task_done()
            cnt += 1

def bootstrap(M, n_samples, query_list, click_set, n_workers):
    task_queue = mp.JoinableQueue()
    manager = mp.Manager()
    res_list = manager.list()

    workers = []
    for _ in range(n_workers):
        w = EstWorker(task_queue, M, click_set, res_list)
        w.daemon = True
        w.start()
        workers.append(w)

    for _ in range(n_samples):
        sample = random.choices(query_list, k=len(query_list))
        task_queue.put(sample)
    for _ in range(n_workers):
        task_queue.put(None)
    task_queue.close()
    task_queue.join()

    for w in workers:
        w.join()
    return res_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Propensity Estimation via swap intervention')
    parser.add_argument('-m', type=int, help='max pos to be estimated')
    parser.add_argument('-n', type=int, default=100, help='num of bootstrap samples')
    parser.add_argument('-p', type=float, default=0.95, help='confdence probability')
    parser.add_argument('--n_workers', default=mp.cpu_count(), type=int,
                        help='number of workers')
    parser.add_argument('query_path', help='query path')
    parser.add_argument('click_path', help='click path')
    parser.add_argument('output_path', help='output path')
    args = parser.parse_args()
    start = timeit.default_timer()

    M = args.m
    n_bootstrap = args.n
    n_workers = min(mp.cpu_count(), args.n_workers)
    conf_prop = args.p
    n_samples = args.n
    n_workers = args.n_workers
    query_path = args.query_path
    click_path = args.click_path

    random.seed()
    click_set = set()
    with open(click_path, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t', fieldnames=click_field_name)
        for line in reader:
            if line['format'] == 'abs':
                click_set.add((line['uid'], line['paper']))

    query_list = []
    with open(query_path, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t', quotechar="'", fieldnames=query_field_name)
        for line in reader:
            uid = line['uid']
            num_results = int(line['num_results'])
            if num_results < M:
                continue
            results = line['results'].split('|')[-1]
            query_list.append((uid, results))

    prop_list = bootstrap(M, n_samples, query_list, click_set, n_workers)
    lo = int(n_samples * ((1 - args.p) / 2))
    mi = int(n_samples * 0.5)
    hi = n_samples - lo

    perc_conf = np.zeros((M, 3))
    for i in range(M):
        p = []
        for prop in prop_list:
            p.append(prop[i])
        p.sort()
        perc_conf[i][0] = p[lo]
        perc_conf[i][1] = p[mi]
        perc_conf[i][2] = p[hi]

    makedirs(os.path.dirname(args.output_path))
    np.savetxt(args.output_path, perc_conf)
    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
