import torch
import numpy as np
import sys
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
from model import *
from visual_backprop import *
from torchsummary import summary


device = torch.device("cuda")
net = OusterNet(num_steers=1).to(device)#models.alexnet().features

ds = OusterConcatDataset('../ouster_data/all', 0.05, only_curves=False, future_steer_dist=10, num_steers=3)
dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
net.eval()
c, w, h =ds[0]["image"].shape
print(type(c))
summary(net, input_size=(c, w, h))


with torch.no_grad():
    for data in dataloader:
        #data = next(iter(dataloader))
        images, labels = data["image"], data["steer"]
        print('In: ', images.shape)
        #images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        print('Out: ', outputs.shape)
