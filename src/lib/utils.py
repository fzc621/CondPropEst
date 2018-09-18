# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np

def makedirs(dirname):
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

def prob_test(prob):
    return random.random() <= prob

def read_para(path, dim):
    if os.path.exists(path):
        with open(path, 'r') as fin:
            w = np.loadtxt(fin)
    else:
        w_ = np.random.uniform(-0.1, 0.1, dim)
        w = w_ - np.mean(w_)
        makedirs(os.path.dirname(path))
        np.savetxt(path, w)
    return w

def cal_prob(w, x, r, method):
    if method == 'power':
        eta = np.dot(w, x) + 1
        if eta < -1:
            print(eta)
        return pow(1 / r, eta)
    elif method == 'exp':
        # exponential function
        a = np.dot(w, x) / 3.2
        return np.power(a, r - 1)
    elif method == 'comp':
        a = np.dot(w, x)
        return 1 / (a * (r - 1) + 1)

def avg_rel_err(p, p_):
    return np.mean(np.absolute(1 - p / p_))
