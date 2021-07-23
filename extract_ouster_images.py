#!/usr/bin/python

import rosbag
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
import argparse

bridge = CvBridge()
amb_c = '/lidar_center/ambient_image'
int_c = '/lidar_center/intensity_image'
rng_c = '/lidar_center/range_image'
steer_topic = '/pacmod/parsed_tx/steer_rpt'
autonomy_topic = '/pacmod/as_tx/enabled'

steer_ratio = 14.7
channels = [amb_c, int_c, rng_c] 


class OusterImage(object):
    def __init__(self, ts):
        self.ts = ts
        self.amb = None
        self.rng = None
        self.inten = None
        self.steer = None

    def set_amb(self, amb):
        self.amb = amb

    def set_inten(self, inten):
        self.inten = inten

    def set_rng(self, rng):
        self.rng = rng

    def set_steer(self, steer):
        self.steer = steer

    def image(self):
        if type(self.rng) != type(None) and type(self.amb) != type(None) and type(self.inten) != type(None) and type(self.steer) != type(None):
            img = np.dstack((self.amb, self.inten, self.rng))
            return img, self.steer
        else:
            print("failed")
            return None, None

img_extension = '.png'
parser = argparse.ArgumentParser(description='Extract Ouster Dataset from ROSBAGs')
parser.add_argument('bag_files', type=str, nargs='+', help='list of rosbags to process')
parser.add_argument('--output_dir', type=str, help='output directory containing the extracted data (default path/to/rosbag/)', default=None)
args = parser.parse_args()
for bag_file in args.bag_files:
    imgs = []
    steers = []
    bag = rosbag.Bag(bag_file)
    first = True
    oi = OusterImage(0)
    wheel = 0.0
    autonomous = True
    autonomy_changed = False
    i = 0

    base_dir = args.output_dir if args.output_dir is not None else os.path.dirname(bag_file)
    output_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(bag_file))[0])
    os.mkdir(output_dir)
    for topic, msg, ts in bag.read_messages():
        if topic == autonomy_topic:
            autonomy_changes = autonomous != msg.data
            autonomous = msg.data
            if autonomy_changed and autonomous:
                oi = OusterImage(0)
        if not autonomous:
            if topic == steer_topic:
                oi.set_steer(msg.manual_input / steer_ratio)
            if topic in channels:
                cv_img = bridge.imgmsg_to_cv2(msg)
                if msg.header.stamp.to_nsec() != oi.ts:
                    if not first: 
                        img, steer = oi.image()
                        if type(img) != type(None):
                            #imgs.append(img)
                            cv2.imwrite(os.path.join(output_dir, str(i)+img_extension), img)
                            i += 1
                            steers.append(steer)
                    oi = OusterImage(msg.header.stamp.to_nsec())
                    first = False
            if topic == amb_c:
                oi.set_amb(cv_img)
            elif topic == int_c:
                oi.set_inten(cv_img)
            elif topic == rng_c:
                oi.set_rng(cv_img)

    steers_file = os.path.join(output_dir, 'steers.csv')
    with open(steers_file, 'w') as f:
        f.write('file_name,steer_angle\n')
        for i, steer in enumerate(steers):
            # Store in Range, Intensity, Ambience order
            f.write(str(i)+img_extension + ',' + str(steer) + '\n')
    print("Saved new folder: " + output_dir)
