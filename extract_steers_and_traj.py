#!/usr/bin/python

import rosbag
import os
import sys
import numpy as np
import pandas as pd
import argparse

steer_topic = '/pacmod/parsed_tx/steer_rpt'
autonomy_topic = '/pacmod/as_tx/enabled'
current_pose_topic = '/current_pose'

steer_ratio = 14.7

parser = argparse.ArgumentParser(description='Extract trajectories from ROSBAGs for CL metric evaluation')
parser.add_argument('bag_files', type=str, nargs='+', help='list of rosbags to process')
parser.add_argument('--output_dir', type=str, help='output directory containing the extracted data (default path/to/rosbag/)', default=None)
args = parser.parse_args()

for bag_file in args.bag_files:
    auts = []
    gps_auts = []
    steers = []
    xs = []
    ys = []
    bag = rosbag.Bag(bag_file)
    print('opened bag: ' + bag_file)
    aut = False

    for topic, msg, ts in bag.read_messages(topics=[steer_topic, autonomy_topic, current_pose_topic]):
        if topic == current_pose_topic:
            xs.append(msg.pose.position.x)
            ys.append(msg.pose.position.y)
            gps_auts.append(aut)
        if topic == autonomy_topic:
            aut = msg.data
        if topic == steer_topic:
            steer = msg.manual_input / steer_ratio
            auts.append(aut)
            steers.append(steer)

    base_dir = args.output_dir if args.output_dir is not None else os.path.dirname(bag_file)
    output_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(bag_file))[0])
    os.mkdir(output_dir)

    steers_file = os.path.join(output_dir, 'steers.csv')
    traj_file = os.path.join(output_dir, 'traj.csv')
    xs_s = pd.Series(xs, name='X')
    ys_s = pd.Series(ys, name='Y')
    gps_auts_s = pd.Series(gps_auts, name='autonomy')
    df_traj = pd.concat([xs_s, ys_s, gps_auts_s], axis=1)
    df_traj.to_csv(traj_file)
    steers_s = pd.Series(steers, name='steer_angle')
    auts_s = pd.Series(auts, name='autonomy')
    df_steer = pd.concat([steers_s, auts_s], axis=1)
    df_steer.to_csv(steers_file)
    print("Saved new folder: " + output_dir)

