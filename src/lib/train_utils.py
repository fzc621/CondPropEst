import copy
import math
import random
import numpy as np
import scipy.optimize as opt

def data_iterator(data, batch_size, shuffle=True):
    idx = range(len(data))
    if shuffle:
        random.shuffle(data)

    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        print('Batch [{}, {}]'.format(start_idx, end_idx))
        yield data[start_idx: end_idx]

def train_nonlinear(M, c, not_c, data_, D, h):
    data = copy.deepcopy(data_)
    def r_idx(k, k_):
        assert k != k_
        if k < k_:
            return (k - 1) * M + k_ - 1
        else:
            return (k_ - 1) * M + k - 1


    a, b = 1e-4, 1-1e-4
    x0 = np.array([random.random() * (b - a) + a] * (M * M + D))
    bnds = [(a, b)] * (M * M + D)

    def likelihood(x):
        theta = x[M * M:]
        r = 0
        for q in data:
            qid = q._qid
            feat = q._feat
            for k in range(1, M + 1):
                for k_ in range(1, M + 1):
                    if k != k_:
                        r += c[(k, k_, qid)] * math.log(h(theta, k, feat) * x[r_idx(k, k_)])
                        r += not_c[(k, k_, qid)] * math.log(1 - h(theta, k, feat) * x[r_idx(k, k_)])
        return -r

    ret = opt.minimize(likelihood, x0, method='L-BFGS-B', bounds=bnds)
    x0 = ret.x

    theta_ = ret.x[M * M:]
    return theta_
