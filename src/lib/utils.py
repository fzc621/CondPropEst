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

def read_para(path):
    if os.path.exists(path):
        with open(path, 'r') as fin:
            w = np.loadtxt(fin)
    else:
        w = -np.random.power(0.5, 10)
        makedirs(os.path.dirname(path))
        np.savetxt(path, w)
    return w

def cal_prob(w, x, r, method):
    if method == 'power':
        eta = np.dot(w, x)
        return pow(1 / r, eta)
    elif method == 'exp':
        # exponential function
        a = np.dot(w, x) / 3.2
        return np.power(a, r - 1)

def _MSE(p, p_):
    return np.sqrt(np.mean(((p - p_)/p_) ** 2))
