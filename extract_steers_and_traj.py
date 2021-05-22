#!/usr/bin/python

import rosbag
import os
import sys
import numpy as np
import pandas as pd
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()
amb_c = '/lidar_center/ambient_image'
int_c = '/lidar_center/intensity_image'
rng_c = '/lidar_center/range_image'
steer_topic = '/pacmod/parsed_tx/steer_rpt'
autonomy_topic = '/pacmod/as_tx/enabled'
current_pose_topic = '/current_pose'

steer_ratio = 14.7
channels = [amb_c, int_c, rng_c]
bag_files = sys.argv[1:]


class OusterImage(object):
    def __init__(self, ts):
        self.ts = ts
        self.amb = None
        self.rng = None
        self.inten = None
        self.steer = None
        self.aut = None

    def set_amb(self, amb):
        self.amb = amb

    def set_inten(self, inten):
        self.inten = inten

    def set_rng(self, rng):
        self.rng = rng

    def set_steer(self, steer):
        self.steer = steer

    def set_aut(self, aut):
        self.aut = aut

    def image(self):
        if all(v is not None for v in [self.rng, self.amb, self.inten, self.steer, self.aut]):
            return self.steer, self.aut
        else:
            print("failed")
            return None, None


for bag_file in bag_files:
    auts = []
    gps_auts = []
    steers = []
    xs = []
    ys = []
    print('opening', bag_file)
    bag = rosbag.Bag(bag_file)
    print('opened bag. reading...')
    first = True
    oi = OusterImage(0)
    aut = False
    i = 0

    for topic, msg, ts in bag.read_messages():
        if topic == current_pose_topic:
            xs.append(msg.pose.position.x)
            ys.append(msg.pose.position.y)
            gps_auts.append(aut)
        if topic == autonomy_topic:
            aut = msg.data
            oi.set_aut(aut)
        if topic == steer_topic:
            oi.set_steer(msg.manual_input / steer_ratio)
        if topic in channels:
            cv_img = bridge.imgmsg_to_cv2(msg)
            if msg.header.stamp.to_nsec() != oi.ts:
                if not first: 
                    steer, aut = oi.image()
                    if type(steer) != type(None):
                        auts.append(aut)
                        steers.append(steer)
                oi = OusterImage(msg.header.stamp.to_nsec())
                first = False
        if topic == amb_c:
            oi.set_amb(cv_img)
        elif topic == int_c:
            oi.set_inten(cv_img)
        elif topic == rng_c:
            if i % 100 == 0:
                print('range', i)
            i += 1
            oi.set_rng(cv_img)

    os.mkdir(bag_file[:-4])
    print('here')
    steers_file = os.path.join(bag_file[:-4], 'steers.csv')
    traj_file = os.path.join(bag_file[:-4], 'traj.csv')
    img_extension = '.png'
    xs_s = pd.Series(xs, name='X')
    ys_s = pd.Series(ys, name='Y')
    gps_auts_s = pd.Series(gps_auts, name='autonomy')
    df_traj = pd.concat([xs_s, ys_s, gps_auts_s], axis=1)
    print('here2')
    df_traj.to_csv(traj_file)
    steers_s = pd.Series(steers, name='steer_angle')
    auts_s = pd.Series(auts, name='autonomy')
    df_steer = pd.concat([steers_s, auts_s], axis=1)
    df_steer.to_csv(steers_file)
    print("Saved new folder: " + bag_file[:-4])
