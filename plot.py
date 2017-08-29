#!/usr/bin/python3


'''
Plots metrics gathered during training.
'''


import numpy as np
from matplotlib import pyplot as plt
from six.moves import cPickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-f', '--full', action='count', help='Show full metrics instead of the mean')
args = parser.parse_args()


with open(args.filename, 'rb') as f: metrics = cPickle.load(f)
side = 4
fig, sps = plt.subplots(side, side)
stack_history = [np.stack(a) for a in zip(*metrics['history'])]
for i in range(len(stack_history)):
    sp = sps[i // side, i % side]
    h = stack_history[i]
    if h.ndim == 1: pl = h
    elif args.full:
        if h.ndim > 2: h = h.mean(tuple(range(1, h.ndim - 1)))
        pl = h.reshape((h.shape[0], -1))
    else: pl = h.mean(tuple(range(1, h.ndim)))
    sp.plot(pl)
    sp.set_title(metrics['names'][i])
plt.show()
