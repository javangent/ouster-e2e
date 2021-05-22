#!/usr/bin/python

import rosbag
import rospy
from sensor_msgs.msg import Image
from ouster_ros.msg import PacketMsg
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
lidar_packets_topic = '/lidar_center/lidar_packets'

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


future_rng = None

def rng_img_callback(msg):
    global future_rng
    future_rng = msg

rospy.init_node('fix_ouster_images', anonymous=True)
pub = rospy.Publisher(lidar_packets_topic, PacketMsg)
sub = rospy.Subscriber(rng_c, Image, rng_img_callback, queue_size=1)

imgs = []
steers = []
for bag_file in bag_files:
    bag = rosbag.Bag(bag_file)
    first = True
    oi = OusterImage(0)
    wheel = 0.0
    autonomous = True
    autonomy_changed = False

    for topic, msg, ts in bag.read_messages():
        if topic == lidar_packets_topic:
            rospy.Rate(1400).sleep()
            pub.publish(msg)
        if topic == autonomy_topic:
            autonomy_changes = autonomous != msg.data
            autonomous = msg.data
            if autonomy_changed and autonomous:
                oi = OusterImage(0)
        if not autonomous:
            if topic == steer_topic:
                oi.set_steer(msg.manual_input / steer_ratio)
            if topic in channels:
                if msg.header.stamp.to_nsec() != oi.ts:
                    if not first: 
                        img, steer = oi.image()
                        if type(img) != type(None):
                            imgs.append(img)
                            steers.append(steer)
                    oi = OusterImage(msg.header.stamp.to_nsec())
                    first = False
            if topic == amb_c:
                cv_img = bridge.imgmsg_to_cv2(msg)
                oi.set_amb(cv_img)
            elif topic == int_c:
                cv_img = bridge.imgmsg_to_cv2(msg)
                oi.set_inten(cv_img)
            elif topic == rng_c:
                for t in range(3000):
                    if future_rng is None:
                        rospy.Rate(1000).sleep()
                        continue
                assert future_rng is not None
                msg.data = future_rng.data
                future_rng = None
                cv_img = bridge.imgmsg_to_cv2(msg)
                oi.set_rng(cv_img)

    os.mkdir(bag_file[:-4])
    steers_file = os.path.join(bag_file[:-4], 'steers.csv')
    img_extension = '.png'
    with open(steers_file, 'w') as f:
        f.write('file_name,steer_angle\n')
        for i, img in enumerate(imgs):
            cv2.imwrite(os.path.join(bag_file[:-4], str(i)+img_extension), img)
            f.write(str(i)+img_extension + ',' + str(steers[i]) + '\n')
    print("Saved new folder: " + bag_file[:-4])
