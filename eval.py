import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import sys
import torch.nn.functional as F
import cv2

def vis(img, label, pred):
    image = img.permute(1,2,0).cpu().data.numpy()
    poff = int((256 - label/(2*np.pi)*1024).cpu().data)
    start_point = (poff, image.shape[1])
    end_point = (poff, 0)
    color = (255, 255, 0)
    thickness = 3
    image = cv2.line(np.ascontiguousarray(image), start_point, end_point, color, thickness) 
    poff = int((256 - pred/(2*np.pi)*1024).cpu().data)
    start_point = (poff, image.shape[1])
    end_point = (poff, 0)
    color = (255, 0, 255)
    thickness = 2
    image = cv2.line(np.ascontiguousarray(image), start_point, end_point, color, thickness) 
    cv2.imshow('img', image)
    cv2.waitKey(200)

model_path, dataset = sys.argv[1:]

with np.load(dataset) as datafile:
    imgs = datafile['images']
    steers = datafile['steering_angles']

print('Dataset file loaded')
train_X = torch.Tensor(imgs).permute(0, 3, 1, 2)[:, :, :, 256:-256]
train_y = torch.Tensor(steers).view(-1, 1)
print(len(train_X))

dataset = TensorDataset(train_X, train_y)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)

net = torch.jit.load(model_path)
net.eval()


with torch.no_grad():
    for data in dataloader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        residuals = (labels - outputs).abs()/np.pi * 180
        sorted_is = residuals.argsort(0, True)[:20]
        print("Max offset (deg): ", residuals[sorted_is[0]])

        for i_max in sorted_is:
            vis(images[int(i_max)], labels[int(i_max),0], outputs[int(i_max),0])
        
