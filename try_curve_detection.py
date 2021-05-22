import torch
import numpy as np
import sys


with np.load(sys.argv[1]) as datafile:
    imgs = datafile['images']
    steers = datafile['steering_angles']
    inds = np.where(np.abs(steers) > 0.015)[0]
    print(inds)
    print(len(inds)/steers.shape[0])

