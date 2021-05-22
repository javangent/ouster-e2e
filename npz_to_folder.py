#!/usr/bin/python

import os
import sys
import numpy as np
import cv2

npz_files = sys.argv[1:]

for npz_file in npz_files:
    with np.load(npz_file) as datafile:
        imgs = datafile['images']
        steers = datafile['steering_angles']

        os.mkdir(npz_file[:-4])
        steers_file = os.path.join(npz_file[:-4], 'steers.csv')
        img_extension = '.png'
        with open(steers_file, 'w') as f:
            f.write('file_name,steer_angle\n')
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(npz_file[:-4], str(i)+img_extension), img)
                f.write(str(i)+img_extension + ',' + str(steers[i]) + '\n')
    print("Saved new folder: " + npz_file[:-4])
