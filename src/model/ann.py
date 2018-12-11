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
        description='propensity estimation w/o condition')
    parser.add_argument('-m', default=10, type=int,
        help='number of top positions for which estimates are desired')
    parser.add_argument('-d', default=10, type=int,
        help='dimension of feature')
    parser.add_argument('-i', '--inference_version', default=0, type=int,
        help='inference version')
    parser.add_argument('-n1', default=12, type=int,
        help='number of propensity hidden layer')
    parser.add_argument('-n2', default=11, type=int,
        help='number of relevance hidden layer')
    parser.add_argument('--gt_dir', help='ground truth directory')
    parser.add_argument('dataset', help='train/valid/test mode')
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
        if args.dataset == 'train':
            train_click_path = os.path.join(args.npy_dir, 'train.click.npy')
            train_c, train_not_c = np.load(train_click_path)
            train_feat_npy_path = os.path.join(args.npy_dir, 'train.feat.npy')
            X_train = np.load(train_feat_npy_path)
            train_gt_path = os.path.join(args.gt_dir, 'set1bin.train.prop.txt')
            Y_train = load_prop(train_gt_path)

            if tf.train.get_checkpoint_state(args.model_dir):
                model.saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))
            else:
                tf.global_variables_initializer().run()

            best_err = math.inf
            for epoch in range(1000):
                p_, train_loss, train_err, _ = sess.run([model.norm_p_, model.loss, model.err, model.train_op],
                         feed_dict={model.x:X_train, model.p:Y_train, model.c:train_c, model.not_c: train_not_c})
                if train_err < best_err:
                    best_err = train_err
                    best_p_ = p_
                    model.saver.save(sess, '{}/checkpoint'.format(args.model_dir), global_step=model.global_step)
                if epoch % 5 == 0:
                    print('{}\tTrain Loss: {:.4f}\tError: {:.4f}\tLowest Error:{:.4f}'
                            .format(epoch, train_loss, train_err, best_err))
            print('Relative Error on training set: {}'.format(best_err))
            prop_path = os.path.join(args.model_dir, 'set1bin.{}.prop.txt'.format(args.dataset))
            np.savetxt(prop_path, p_, fmt='%.18f')
        else:
            if args.inference_version == 0:  # Load the checkpoint
                model_path = tf.train.latest_checkpoint(args.model_dir)
            else:
                model_path = '{}/checkpoint-{}'.format(args.model_dir, args.inference_version)
            model.saver.restore(sess, model_path)

            test_feat_npy_path = os.path.join(args.npy_dir, '{}.feat.npy'.format(args.dataset))
            X_test = np.load(test_feat_npy_path)
            test_gt_path = os.path.join(args.gt_dir, 'set1bin.{}.prop.txt'.format(args.dataset))
            Y_test = load_prop(test_gt_path)

            p_, test_err = sess.run([model.norm_p_, model.err],
                                    feed_dict={model.x:X_test, model.p:Y_test})
            print('Relative Error on {} set: {}'.format(args.dataset, test_err))
            test_prop_path = os.path.join(args.model_dir, 'set1bin.{}.prop.txt'.format(args.dataset))
            np.savetxt(test_prop_path, p_, fmt='%.18f')
    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
