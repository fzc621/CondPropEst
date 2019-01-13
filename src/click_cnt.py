# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='click cnt')
    parser.add_argument('log_path', help='click log path')

    args = parser.parse_args()
    start = timeit.default_timer()
    cnt = 0
    with open(args.log_path) as fin:
        for line in fin:
            cnt += 1
    print(cnt)
    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
