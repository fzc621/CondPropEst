# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import timeit
import numpy as np
import argparse
import multiprocessing as mp
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
            global_c, global_not_c = np.zeros((M, M)), np.zeros((M, M))
            for c, not_c in query_set:
                global_c += c
                global_not_c += not_c
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
            res_list.append(prop_)
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
    parser.add_argument('-n', type=int, default=1000, help='num of bootstrap samples')
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
            click_set.add((line['uid'], line['paper']))

    query_list = []
    with open(query_path, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE,
                                fieldnames=query_field_name)
        for row in reader:
            uid = row['uid']
            toks = row['results'].split('*')
            all_ranks = []
            for i in range(3):
                all_ranks.append(toks[3 + i].split(',')[1:])
            c, not_c = np.zeros((M, M)), np.zeros((M, M))
            selected_ranker_id = int(toks[1])
            selected_ranker = all_ranks[selected_ranker_id]
            selected_length = len(selected_ranker)
            max_length = min(selected_length, M)
            for k in range(max_length):
                doc = selected_ranker[k]
                weight = 0
                for i in range(3):
                    ranker = all_ranks[i]
                    if len(ranker) <= k:
                        continue
                    if ranker[k] == doc:
                        weight += 1

                for k_ in range(max_length):
                    if k_ == k:
                        continue

                    for i in range(3):
                        ranker = all_ranks[i]
                        if len(ranker) <= k_:
                            continue
                        if ranker[k_] == doc:
                            if (uid, doc) in click_set:
                                c[k][k_] += 1.0 / weight
                            else:
                                not_c[k][k_] += 1.0 / weight
            query_list.append((c, not_c))

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
