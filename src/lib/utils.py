# -*- coding: utf-8 -*-

import os
import sys
import random
import parse
import numpy as np

def makedirs(dirname):
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

def prob_test(prob):
    return random.random() <= prob

def read_para(path, dim, range):
    if os.path.exists(path):
        with open(path, 'r') as fin:
            w = np.loadtxt(fin)
            if w[0] == range:
                return w[1:]
    w_ = np.random.uniform(-range, range, dim)
    w = w_ - np.mean(w_)
    print('norm(w) = {}'.format(np.linalg.norm(w)))
    makedirs(os.path.dirname(path))
    x = np.hstack((range, w))
    np.savetxt(path, x)
    return w

def cal_prob(w, x, r, method):
    if method == 'power':
        eta = np.dot(w, x) + 1
        assert eta >= 0
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

def read_test_err(path):
    with open(path) as fin:
        line = fin.readline().rstrip()
        err = parse.parse('Relative Error on test set: {}', line)
        return float(err[0])
