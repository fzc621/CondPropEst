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

def read_err(dir, dataset):
    path = os.path.join(dir, '{}.txt'.format(dataset))
    with open(path) as fin:
        line = fin.readline().rstrip()
        err = parse.parse('Relative Error on {} set: '.format(dataset) + '{}', line)
        return float(err[0])

def find_best_prop_model(dir):
    best_path = None
    best_err = 10e99
    for n in os.listdir(dir):
        if not n.startswith('.'):
            path = os.path.join(dir, n)
            err = read_err(path, 'valid')
            if err < best_err:
                best_err = err
                best_path = path
    return best_path

def find_best_rel_model(dir):
    best_path = None
    best_err = 10e99
    for n1 in os.listdir(dir):
        if not n1.startswith('.'):
            n1_path = os.path.join(dir, n1)
            for n2 in os.listdir(n1_path):
                if not n2.startswith('.'):
                    path = os.path.join(n1_path, n2)
                    err = read_err(path, 'valid')
                    if err < best_err:
                        best_err = err
                        best_path = path
    return best_path
