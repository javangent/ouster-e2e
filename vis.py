import numpy as np
import os
from collections import deque
import sys
import cv2
from data import OusterConcatDataset

def draw_overlay(img, steer, ind, curves):
    fov = 256
    left, right, top, bottom = 512-fov//2,512+fov//2,62,128-5
    color = (0,255,0)
    thickness = 2
    
    img_overlay = cv2.line(img, (left, top), (left, bottom), color, thickness)
    img_overlay = cv2.line(img_overlay, (right, top), (right, bottom), color, thickness)
    img_overlay = cv2.line(img_overlay, (left, top), (right, top), color, thickness)
    img_overlay = cv2.line(img_overlay, (left, bottom), (right, bottom), color, thickness)

#    img_overlay = cv2.putText(img, str(ind), (0,img.shape[0]), cv2.FONT_HERSHEY_SIMPLEX ,
#       1, (255,255,255), 1, cv2.LINE_AA)
#
#    if ind in curves:
#        img_overlay = cv2.putText(img_overlay, 'CURVE', (0, 20), cv2.FONT_HERSHEY_SIMPLEX ,
#       1, (255,255,0), 2, cv2.LINE_AA)
#
#    img_overlay = cv2.putText(img_overlay, '{:.2f}'.format(steer[0]/np.pi*180), (0, 60), cv2.FONT_HERSHEY_SIMPLEX ,
#       1, (255,255,255), 1, cv2.LINE_AA)
#
#    ouster_offset = 15
#
#    x = img.shape[1]//2 + ouster_offset
#    start = (x, 0)
#    end = (x, img.shape[0])
#    color = (0, 0, 255)
#    thickness = 3
#    img_overlay = cv2.line(img_overlay, start, end, color, thickness)
#
#    for s in steer:
#        x = -int(s / (2 * np.pi) * 1024) + 512 + ouster_offset
#        start = (x, 0)
#        end = (x, img.shape[0])
#        color = (0, 255, 0)
#        thickness = 2
#        img_overlay = cv2.line(img_overlay, start, end, color, thickness)

    return img_overlay

folder = sys.argv[1]
ds = OusterConcatDataset(folder, 0.05, only_curves=False, frame_dist=0, future_steer_dist=10, num_steers=3, image_format='ria')
curves = ds.get_curve_inds()
deq = deque(range(0, len(ds)))
img, label, idx = ds.get_numpy_sample(deq[0])
print(len(label))
img_overlay = draw_overlay(img, label, deq[0], curves)
image_rgb = cv2.cvtColor(img_overlay,cv2.COLOR_BGR2RGB)
cv2.imshow('Dataset Viewer',image_rgb)

while cv2.getWindowProperty('Dataset Viewer', cv2.WND_PROP_VISIBLE) >= 1:
    image_rgb = cv2.cvtColor(img_overlay,cv2.COLOR_BGR2RGB)
    cv2.imshow('Dataset Viewer',image_rgb)
    k = cv2.waitKey(2)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        break
    elif k == ord('j'): # wait for 'j' key to change to next image
        deq.rotate(1)
        img, label, idx = ds.get_numpy_sample(deq[0])
        img_overlay = draw_overlay(img, label, deq[0], curves)
    elif k == ord('k'): # wait for 'k' key to change to previous image
        deq.rotate(-1)
        img, label, idx = ds.get_numpy_sample(deq[0])
        img_overlay = draw_overlay(img, label, deq[0], curves)

