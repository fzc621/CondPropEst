# -*- coding: utf-8 -*-

import os
import sys
import timeit
import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.style.use('classic')

def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None, label=None):
    plt.fill_between(range(1, 22), ub, lb,
                     color=color_shading, alpha=.3)
    plt.plot(range(1, 22), mean, color_mean, linewidth=2, label=label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot')
    parser.add_argument('model_dir', help='model dir')
    parser.add_argument('output_path', help='output path')

    args = parser.parse_args()
    start = timeit.default_timer()

    simple_prop_path = os.path.join(args.model_dir, 'simple_bootstrap.txt')
    simple_prop = np.loadtxt(simple_prop_path).transpose()
    complex_prop_path = os.path.join(args.model_dir, 'complex_bootstrap.txt')
    complex_prop = np.loadtxt(complex_prop_path).transpose()

    plt.figure()

    plot_mean_and_CI(simple_prop[1], simple_prop[0], simple_prop[2],
                     color_mean='g', color_shading='g',
                     label='Simple Queries')
    plot_mean_and_CI(complex_prop[1], complex_prop[0], complex_prop[2],
                     color_mean='b', color_shading='b',
                     label='Complex Queries')
    plt.xticks(range(0, 22), range(0, 22))
    plt.xlabel('Position')
    plt.ylabel('Relative Propensity')
    plt.legend(frameon=False, markerfirst=False)
    _, _, _, y1 = plt.axis()
    plt.axis((1, 21, 0, y1))
    plt.savefig(args.output_path)
    # plt.show()

    end = timeit.default_timer()
    print('Runing time: {:.3f}s.'.format(end - start))
