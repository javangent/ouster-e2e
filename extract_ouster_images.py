#!/usr/bin/python

import rosbag
import os
import sys
import numpy as np
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()
amb_c = '/lidar_center/ambient_image'
int_c = '/lidar_center/intensity_image'
rng_c = '/lidar_center/range_image'
steer_topic = '/pacmod/parsed_tx/steer_rpt'
autonomy_topic = '/pacmod/as_tx/enabled'

steer_ratio = 14.7
channels = [amb_c, int_c, rng_c]
top = 50
left = 0 #+ 256
right = 1024 #- 256
bottom = 128 - 5
bag_files = sys.argv[1:]


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


for bag_file in bag_files:
    imgs = []
    steers = []
    bag = rosbag.Bag(bag_file)
    first = True
    oi = OusterImage(0)
    wheel = 0.0
    autonomous = True
    autonomy_changed = False

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
                            imgs.append(img)
                            steers.append(steer)
                    oi = OusterImage(msg.header.stamp.to_nsec())
                    first = False
            if topic == amb_c:
                oi.set_amb(cv_img)
            elif topic == int_c:
                oi.set_inten(cv_img)
            elif topic == rng_c:
                oi.set_rng(cv_img)

    os.mkdir(bag_file[:-4])
    steers_file = os.path.join(bag_file[:-4], 'steers.csv')
    img_extension = '.png'
    with open(steers_file, 'w') as f:
        f.write('file_name,steer_angle\n')
        for i, img in enumerate(imgs):
            # Store in Range, Intensity, Ambience order
            cv2.imwrite(os.path.join(bag_file[:-4], str(i)+img_extension), img)
            f.write(str(i)+img_extension + ',' + str(steers[i]) + '\n')
    print("Saved new folder: " + bag_file[:-4])
