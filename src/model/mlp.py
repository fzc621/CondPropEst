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


l, h = 1e-6, 1 - 1e-6
N1 = 16#power:12 exp:16
N2 = 12

def prob_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    # initial = tf.random_uniform(shape, minval=l, maxval=h)
    return tf.Variable(initial)

def weight_variable(shape):
    # initial = tf.constant(0.1, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # initial = tf.constant(0.1, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

class MLP(object):
    def __init__(self, D, M, learning_rate=3e-2):
        self.x = tf.placeholder(tf.float32, shape=(None, D))
        self.p = tf.placeholder(tf.float32, shape=(None, M))
        self.c = tf.placeholder(tf.float32, shape=(None, M, M))
        self.not_c = tf.placeholder(tf.float32, shape=(None, M, M))

        with tf.variable_scope('fc1'):
            w_fc1 = weight_variable([D, N1])
            b_fc1 = bias_variable([N1])
            h_fc1 = tf.nn.relu(tf.matmul(self.x, w_fc1) + b_fc1)

        with tf.variable_scope('fc2'):
            w_fc2 = weight_variable([N1, M])
            b_fc2 = bias_variable([M])
            # h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
            p_ = tf.nn.sigmoid(tf.matmul(h_fc1, w_fc2) + b_fc2)

        # with tf.variable_scope('fc3'):
        #     w_fc3 = weight_variable([N2, M])
        #     b_fc3 = bias_variable([M])
        #     p_ = tf.nn.sigmoid(tf.matmul(h_fc2, w_fc3) + b_fc3)


        r = prob_variable((M, M))
        r_symm = tf.div(tf.add(r, tf.transpose(r)), 2.0)
        clip_r = tf.clip_by_value(r_symm, clip_value_min=l, clip_value_max=h)

        self.pr = tf.reshape(p_,[-1, M, 1]) * clip_r
        self.loss = -tf.reduce_sum(tf.add(self.c * tf.log(self.pr), self.not_c * tf.log(1 - self.pr)))

        self.norm_p_ = tf.div(p_, tf.reshape(p_[:,0], (-1, 1)))
        self.mse = tf.sqrt(tf.reduce_mean(tf.square((self.p - self.norm_p_)) / self.norm_p_))

        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3,
                        pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
