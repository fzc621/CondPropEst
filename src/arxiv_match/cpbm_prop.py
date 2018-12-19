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
from ..model import mlp, mlp_rel
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
    parser.add_argument('dataset', help='train/valid/test mode')
    parser.add_argument('model', choices=['mlp', 'mlp_rel'])
    parser.add_argument('data_dir', help='data dir')
    parser.add_argument('output_dir', help='model directory')
    args = parser.parse_args()

    start = timeit.default_timer()

    M = args.m
    D = args.d

    makedirs(args.output_dir)
    with tf.Session() as sess:
        if args.model == 'mlp':
            model = mlp.MLP(D, M, args.n1)
        elif args.model == 'mlp_rel':
            model = mlp_rel.MLP(D, M, args.n1, args.n2)

        if args.dataset == 'train':
            train_click_path = os.path.join(args.data_dir, 'train.click.npy')
            train_c, train_not_c = np.load(train_click_path)
            train_feat_path = os.path.join(args.data_dir, 'train.feat.npy')
            X_train = np.load(train_feat_path)

            if tf.train.get_checkpoint_state(args.output_dir):
                model.saver.restore(sess, tf.train.latest_checkpoint(args.output_dir))
            else:
                tf.global_variables_initializer().run()

            best_loss = math.inf
            for epoch in range(1500):
                train_loss, _ = sess.run([model.loss, model.train_op],
                        feed_dict={model.x:X_train, model.c:train_c, model.not_c: train_not_c})
                if train_loss < best_loss:
                    best_loss = train_loss
                    model.saver.save(sess, '{}/checkpoint'.format(args.output_dir), global_step=model.global_step)
                if epoch % 100 == 0:
                    print('{}\tTrain Loss: {:.4f}'.format(epoch, train_loss))
                    X = np.array([[0],[1]])
                    p_ = sess.run([model.norm_p_], feed_dict={model.x:X})[0]

            X = np.array([[0],[1]])
            p_ = sess.run([model.norm_p_], feed_dict={model.x:X})[0]
            complex_prop_path = os.path.join(args.output_dir, 'complex_prop.txt')
            np.savetxt(complex_prop_path, p_[1], fmt='%.18f')
            simple_prop_path = os.path.join(args.output_dir, 'simple_prop.txt')
            np.savetxt(simple_prop_path, p_[0], fmt='%.18f')
        else:
            if args.inference_version == 0:  # Load the checkpoint
                model_path = tf.train.latest_checkpoint(args.output_dir)
            else:
                model_path = '{}/checkpoint-{}'.format(args.output_dir, args.inference_version)
            model.saver.restore(sess, model_path)
            test_click_path = os.path.join(args.data_dir, 'valid.click.npy')
            test_c, test_not_c = np.load(test_click_path)
            test_feat_path = os.path.join(args.data_dir, 'valid.feat.npy')
            X_test = np.load(test_feat_path)

            test_loss = sess.run([model.loss], feed_dict={model.x:X_test,
                                model.c:test_c, model.not_c: test_not_c})[0]
            print('Loss on validation set: {}'.format(test_loss))

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
