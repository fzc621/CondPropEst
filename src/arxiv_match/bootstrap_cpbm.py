# -*- coding: utf-8 -*-

import os
import sys
import math
import random
import timeit
import numpy as np
import argparse
import multiprocessing as mp
import tensorflow as tf
from ..model import mlp, mlp_rel
from ..lib.utils import makedirs

class EstWorker(mp.Process):

    def __init__(self, task_queue, D, M, n1, n2, output_dir, res_list):
        super(EstWorker, self).__init__()
        self._task_queue = task_queue
        self._D = D
        self._M = M
        self._n1 = n1
        self._n2 = n2
        self._output_dir = output_dir
        self._res_list = res_list

    def run(self):
        task_queue = self._task_queue
        n1 = self._n1
        n2 = self._n2
        res_list = self._res_list
        D = self._D
        M = self._M
        name = self.name
        output_dir = self._output_dir

        cnt = 0
        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                print('{}: Processed {} tasks'.format(name, cnt))
                break
            feat, c, not_c = task
            task_output_dir = os.path.join(output_dir, '{}_{}'.format(name, cnt))
            makedirs(task_output_dir)
            with tf.Session() as sess:
                model = mlp_rel.MLP(D, M, n1, n2)
                tf.global_variables_initializer().run()
                best_loss = math.inf
                for epoch in range(10):
                    train_loss, _ = sess.run([model.loss, model.train_op],
                            feed_dict={model.x:feat, model.c:c,
                                        model.not_c: not_c})
                    if train_loss < best_loss:
                        best_loss = train_loss
                        model.saver.save(sess, '{}/checkpoint'.format(task_output_dir),
                                        global_step=model.global_step)
                X = np.array([[0],[1]])
                p_ = sess.run([model.norm_p_], feed_dict={model.x:X})[0]
            res_list.append(p_)
            task_queue.task_done()
            cnt += 1

def bootstrap(D, M, n_samples, c, not_c, feat, n1, n2, output_dir, n_workers):
    task_queue = mp.JoinableQueue()
    manager = mp.Manager()
    res_list = manager.list()

    workers = []
    for _ in range(n_workers):
        w = EstWorker(task_queue, D, M, n1, n2, output_dir, res_list)
        w.daemon = True
        w.start()
        workers.append(w)

    data_size = feat.shape[0]
    for _ in range(n_samples):
        data_idx = random.choices(range(data_size), k=data_size)
        b_feat, b_c, b_not_c = [], [], []
        for i in data_idx:
            b_feat.append(feat[i])
            b_c.append(c[i])
            b_not_c.append(not_c[i])
        b_feat = np.array(b_feat).reshape(data_size, -1)
        b_c = np.array(b_c).reshape(data_size, M, M)
        b_not_c = np.array(b_not_c).reshape(data_size, M, M)
        task_queue.put((b_feat, b_c, b_not_c))

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
    parser.add_argument('-d', default=1, type=int, help='dimension of feature')
    parser.add_argument('-n', type=int, default=1000, help='num of bootstrap samples')
    parser.add_argument('-n1', default=32, type=int,
        help='number of propensity hidden layer')
    parser.add_argument('-n2', default=16, type=int,
        help='number of relevance hidden layer')
    parser.add_argument('-p', type=float, default=0.95, help='confidence prob')
    parser.add_argument('--n_workers', default=mp.cpu_count(), type=int,
                        help='number of workers')
    parser.add_argument('data_dir', help='data dir')
    parser.add_argument('mlp_dir', help='mlp dir')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()
    start = timeit.default_timer()


    M = args.m
    D = args.d
    n_bootstrap = args.n
    n_workers = min(mp.cpu_count(), args.n_workers)
    conf_prop = args.p
    n_samples = args.n
    n_workers = args.n_workers
    n1, n2 = args.n1, args.n2
    output_dir = args.output_dir

    random.seed()
    click_path = os.path.join(args.data_dir, 'train.click.npy')

    c, not_c = np.load(click_path)
    feat_path = os.path.join(args.data_dir, 'train.feat.npy')
    feat = np.load(feat_path)
    prop_list = bootstrap(D, M, n_samples, c, not_c, feat, n1, n2,
                            output_dir, n_workers)
    lo = int(n_samples * ((1 - args.p) / 2))
    mi = int(n_samples * 0.5)
    hi = n_samples - lo

    perc_conf = np.zeros((2, M, 3))
    for i in range(2):
        for j in range(M):
            p = []
            for prop in prop_list:
                p.append(prop[i][j])
            p.sort()
            perc_conf[i][j][0] = p[lo]
            perc_conf[i][j][1] = p[mi]
            perc_conf[i][j][2] = p[hi]

    makedirs(args.output_dir)
    simple_prop_path = os.path.join(args.output_dir, 'simple_bootstrap.txt')
    np.savetxt(simple_prop_path, perc_conf[0], fmt='%.18f')
    complex_prop_path = os.path.join(args.output_dir, 'complex_bootstrap.txt')
    np.savetxt(complex_prop_path, perc_conf[1], fmt='%.18f')

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
