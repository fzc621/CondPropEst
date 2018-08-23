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

def likelihood(p, r, c, not_c, M):
    pr = np.repeat(p, M).reshape([M, M]) * r
    obj = np.sum(c * np.log10(pr) + not_c * np.log10(1 - pr))
    return obj
