# -*- coding: utf-8 -*-

import os
import sys
import math
import copy
import random
import timeit
import argparse
import numpy as np
import tensorflow as tf
from . import mlp, mlp_rel
from ..lib.data_utils import load_prop
from ..lib.utils import makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='propensity estimation w/o condition for arxiv')
    parser.add_argument('-m', default=21, type=int,
        help='number of top positions for which estimates are desired')
    parser.add_argument('-d', default=1, type=int,
        help='dimension of feature')
    parser.add_argument('-i', '--inference_version', default=0, type=int,
        help='inference version')
    parser.add_argument('-n1', default=12, type=int,
        help='number of propensity hidden layer')
    parser.add_argument('-n2', default=11, type=int,
        help='number of relevance hidden layer')
    parser.add_argument('model', choices=['mlp', 'mlp_rel'])
    parser.add_argument('npy_dir', help='numpy dir')
    parser.add_argument('model_dir', help='model directory')
    args = parser.parse_args()

    start = timeit.default_timer()

    M = args.m
    D = args.d

    makedirs(args.model_dir)
    with tf.Session() as sess:
        if args.model == 'mlp':
            model = mlp.MLP(D, M, args.n1)
        elif args.model == 'mlp_rel':
            model = mlp_rel.MLP(D, M, args.n1, args.n2)

        train_click_path = os.path.join(args.npy_dir, 'train.click.npy')
        train_c, train_not_c = np.load(train_click_path)
        train_feat_npy_path = os.path.join(args.npy_dir, 'train.feat.npy')
        X_train = np.load(train_feat_npy_path)

        if tf.train.get_checkpoint_state(args.model_dir):
            model.saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))
        else:
            tf.global_variables_initializer().run()

        best_loss = math.inf
        for epoch in range(10):
            train_loss, _ = sess.run([model.loss, model.train_op],
                    feed_dict={model.x:X_train, model.c:train_c, model.not_c: train_not_c})
            if train_loss < best_loss:
                best_loss = train_loss
                model.saver.save(sess, '{}/checkpoint'.format(args.model_dir), global_step=model.global_step)
            if epoch % 5 == 0:
                print('{}\tTrain Loss: {:.4f}'.format(epoch, train_loss))

        X = np.array([[0],[1]])
        p_ = sess.run([model.norm_p_], feed_dict={model.x:X})[0]
        print(p_)
        complex_prop_path = os.path.join(args.model_dir, 'complex_prop.txt')
        np.savetxt(complex_prop_path, p_[0], fmt='%.18f')
        simple_prop_path = os.path.join(args.model_dir, 'simple_prop.txt')
        np.savetxt(simple_prop_path, p_[1], fmt='%.18f')

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
