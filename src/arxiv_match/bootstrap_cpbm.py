# -*- coding: utf-8 -*-

import os
import sys
import math
import random
import timeit
import numpy as np
import argparse
import tensorflow as tf
from ..model import mlp, mlp_rel
from ..lib.utils import makedirs

def bootstrap(D, M, c, not_c, feat, n1, n2, output_dir):
    data_size = feat.shape[0]
    data_idx = random.choices(range(data_size), k=data_size)
    b_feat, b_c, b_not_c = [], [], []
    for i in data_idx:
        b_feat.append(feat[i])
        b_c.append(c[i])
        b_not_c.append(not_c[i])
    bs_feat = np.array(b_feat).reshape(data_size, -1)
    bs_c = np.array(b_c).reshape(data_size, M, M)
    bs_not_c = np.array(b_not_c).reshape(data_size, M, M)
    del b_feat, b_c, b_not_c

    makedirs(output_dir)
    with tf.Session() as sess:
        model = mlp_rel.MLP(D, M, n1, n2)
        tf.global_variables_initializer().run()
        best_loss = math.inf
        for epoch in range(2000):
            train_loss, _ = sess.run([model.loss, model.train_op],
                    feed_dict={model.x:bs_feat, model.c:bs_c,
                                model.not_c: bs_not_c})
            if train_loss < best_loss:
                best_loss = train_loss
                model.saver.save(sess, '{}/checkpoint'.format(output_dir),
                                global_step=model.global_step)
        X = np.array([[0],[1]])
        p_ = sess.run([model.norm_p_], feed_dict={model.x:X})[0]
    return p_

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Propensity Estimation via swap intervention')
    parser.add_argument('-m', type=int, help='max pos to be estimated')
    parser.add_argument('-d', default=1, type=int, help='dimension of feature')
    parser.add_argument('-n1', default=32, type=int,
        help='number of propensity hidden layer')
    parser.add_argument('-n2', default=16, type=int,
        help='number of relevance hidden layer')
    parser.add_argument('data_dir', help='data dir')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()
    start = timeit.default_timer()


    M = args.m
    D = args.d
    n1, n2 = args.n1, args.n2
    output_dir = args.output_dir

    random.seed()
    click_path = os.path.join(args.data_dir, 'train.click.npy')

    c, not_c = np.load(click_path)
    feat_path = os.path.join(args.data_dir, 'train.feat.npy')
    feat = np.load(feat_path)
    prop = bootstrap(D, M, c, not_c, feat, n1, n2, output_dir)

    makedirs(args.output_dir)
    simple_prop_path = os.path.join(args.output_dir, 'simple_bootstrap.txt')
    np.savetxt(simple_prop_path, prop[0], fmt='%.18f')
    complex_prop_path = os.path.join(args.output_dir, 'complex_bootstrap.txt')
    np.savetxt(complex_prop_path, prop[1], fmt='%.18f')

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
