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
from .mlp import MLP
from ..lib.data_utils import load_prop
from ..lib.utils import makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='propensity estimation w/o condition')
    parser.add_argument('-m', default=10, type=int,
        help='number of top positions for which estimates are desired')
    parser.add_argument('-d', default=10, type=int,
        help='dimension of feature')
    parser.add_argument('-e', '--epoch_num',default=600, type=int,
        help='#epoch')
    parser.add_argument('-i', '--inference_version', default=0, type=int,
        help='inference version')
    parser.add_argument('--test', action='store_true', help='train/test mode')
    parser.add_argument('--gt_dir', help='ground truth directory')
    parser.add_argument('--log_dir', help='click log dir')
    parser.add_argument('model', choices=['mlp'])
    parser.add_argument('npy_dir', help='numpy dir')
    parser.add_argument('model_dir', help='model directory')
    args = parser.parse_args()

    start = timeit.default_timer()

    M = args.m
    D = args.d

    makedirs(args.model_dir)
    with tf.Session() as sess:
        if args.model == 'mlp':
            model = MLP(D, M)
        if not args.test:
            click_npy_path = os.path.join(args.npy_dir, 'click.info.npy')
            c, not_c = np.load(click_npy_path)
            train_feat_npy_path = os.path.join(args.npy_dir, 'train.feat.npy')
            X_train = np.load(train_feat_npy_path)
            train_gt_path = os.path.join(args.gt_dir, 'set1bin.train.prop.txt')
            Y_train = load_prop(train_gt_path)

            if tf.train.get_checkpoint_state(args.model_dir):
                model.saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))
            else:
                tf.global_variables_initializer().run()

            best_mse = math.inf
            for epoch in range(args.epoch_num):
                p_, train_loss, train_mse, _ = sess.run([model.norm_p_, model.loss, model.mse, model.train_op],
                         feed_dict={model.x:X_train, model.p:Y_train, model.c:c, model.not_c: not_c})
                if train_mse < best_mse:
                    best_p = p_
                    best_mse = train_mse
                    model.saver.save(sess, '{}/checkpoint'.format(args.model_dir), global_step=model.global_step)
                if epoch % 5 == 0:
                    print('{}\tLOSS: {:.4f}\tMSE:{:.4f}\tBest MSE:{:.4f}'.format(epoch, train_loss, train_mse, best_mse))
        else:
            if args.inference_version == 0:  # Load the checkpoint
                model_path = tf.train.latest_checkpoint(args.model_dir)
            else:
                model_path = '{}/checkpoint-{}'.format(args.model_dir, args.inference_version)
            model.saver.restore(sess, model_path)

            test_feat_npy_path = os.path.join(args.npy_dir, 'test.feat.npy')
            X_test = np.load(test_feat_npy_path)
            test_gt_path = os.path.join(args.gt_dir, 'set1bin.test.prop.txt')
            Y_test = load_prop(test_gt_path)

            p_, test_mse = sess.run([model.norm_p_, model.mse],
                                    feed_dict={model.x:X_test, model.p:Y_test})

            test_prop_path = os.path.join(args.model_dir,
                                        'test.prop.mse{:.3f}.txt'.format(test_mse))

            np.savetxt(test_prop_path, p_, fmt='%.18f')
    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))