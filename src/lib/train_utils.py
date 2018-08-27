import copy
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
