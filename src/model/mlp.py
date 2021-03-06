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
N2 = 12

def prob_variable(shape):
    initial = tf.random_uniform(shape, minval=l, maxval=h)
    return tf.Variable(initial)

def weight_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

class MLP(object):
    def __init__(self, D, M, N1=12, learning_rate=3e-2):
        self.x = tf.placeholder(tf.float32, shape=(None, D))
        self.p = tf.placeholder(tf.float32, shape=(None, M))
        self.c = tf.placeholder(tf.float32, shape=(None, M, M))
        self.not_c = tf.placeholder(tf.float32, shape=(None, M, M))

        with tf.variable_scope('fc1'):
            w_fc1 = weight_variable([D, N1])
            b_fc1 = bias_variable([N1])
            h_fc1 = tf.nn.sigmoid(tf.matmul(self.x, w_fc1) + b_fc1)

        with tf.variable_scope('fc2'):
            w_fc2 = weight_variable([N1, M])
            b_fc2 = bias_variable([M])
            p_ = tf.nn.sigmoid(tf.matmul(h_fc1, w_fc2) + b_fc2)

        r = prob_variable((M, M))
        r_symm = tf.div(tf.add(r, tf.transpose(r)), 2.0)
        clip_r = tf.clip_by_value(r_symm, clip_value_min=l, clip_value_max=h)

        self.pr = tf.reshape(p_,[-1, M, 1]) * clip_r
        self.loss = -tf.reduce_sum(tf.add(self.c * tf.log(self.pr), self.not_c * tf.log(1 - self.pr)))

        self.norm_p_ = tf.div(p_, tf.reshape(p_[:,0], (-1, 1)))
        self.err = tf.reduce_mean(tf.abs((self.p - self.norm_p_) / self.norm_p_))

        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3,
                        pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
