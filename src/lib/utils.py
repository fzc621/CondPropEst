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
        w = np.random.power(0.5, 10)
        np.savetxt(path, w)
    return w

def cal_prob(w, x, r):
    eta = np.dot(w, x)
    return pow(1 / r, eta)

def _MSE(p, p_):
    ip = 1 / p
    ip_ = 1 / p_
    return np.mean((ip - ip_) ** 2)
