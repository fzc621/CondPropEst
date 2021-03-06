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
from ..model import mlp_rel
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
    parser.add_argument('--epoch', default=10000, type=int,
        help='#epoch')
    parser.add_argument('-n1', default=32, type=int,
        help='number of propensity hidden layer')
    parser.add_argument('-n2', default=32, type=int,
        help='number of relevance hidden layer')
    parser.add_argument('feat_type', help='feat type')
    parser.add_argument('data_dir', help='data dir')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()

    start = timeit.default_timer()

    M = args.m
    D = args.d

    makedirs(args.output_dir)
    with tf.Session() as sess:
        model = mlp_rel.MLP(D, M, args.n1, args.n2, 0.1)

        train_click_path = os.path.join(args.data_dir, 'train.click.npy')
        train_c, train_not_c = np.load(train_click_path)
        train_feat_path = os.path.join(args.data_dir, 'train.{}.feat.npy'.format(args.feat_type))

        valid_click_path = os.path.join(args.data_dir, 'valid.click.npy')
        valid_c, valid_not_c = np.load(valid_click_path)
        valid_feat_path = os.path.join(args.data_dir, 'valid.{}.feat.npy'.format(args.feat_type))
        X_valid = np.load(valid_feat_path)
        valid_loss_path = os.path.join(args.output_dir, 'valid_loss.txt')

        test_click_path = os.path.join(args.data_dir, 'test.click.npy')
        test_c, test_not_c = np.load(test_click_path)
        test_feat_path = os.path.join(args.data_dir, 'test.{}.feat.npy'.format(args.feat_type))
        X_test = np.load(test_feat_path)
        test_loss_path = os.path.join(args.output_dir, 'test_loss.txt')

        X_train = np.load(train_feat_path)
        if tf.train.get_checkpoint_state(args.output_dir):
            model.saver.restore(sess, tf.train.latest_checkpoint(args.output_dir))
        else:
            tf.global_variables_initializer().run()

        best_loss = math.inf
        for epoch in range(args.epoch):
            train_loss, _ = sess.run([model.loss, model.train_op],
                    feed_dict={model.x:X_train, model.c:train_c,
                               model.not_c: train_not_c})
            valid_loss = sess.run([model.loss],
                                    feed_dict={model.x:X_valid, model.c:valid_c,
                                                model.not_c: valid_not_c})[0]
            if valid_loss < best_loss:
                best_loss = valid_loss
                model.saver.save(sess, '{}/checkpoint'.format(args.output_dir), global_step=model.global_step)

            if epoch % 100 == 0:
                print('{}\tTrain Loss: {:.4f} Best Valid Loss: {:.4f}'.format(epoch, train_loss, valid_loss))

        model.saver.restore(sess, tf.train.latest_checkpoint(args.output_dir))

        with open(valid_loss_path, 'w') as fout:
            fout.write('Loss: {}'.format(valid_loss))

        test_loss = sess.run([model.loss],
                                feed_dict={model.x:X_test, model.c:test_c,
                                            model.not_c: test_not_c})[0]
        with open(test_loss_path, 'w') as fout:
            fout.write('Loss: {}'.format(test_loss))

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
