import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Subset
from data import OusterConcatDataset
from torch.utils.tensorboard import SummaryWriter
import cv2
from model import *
import pandas as pd
from torchvision import models, io
import os
import argparse

parser = argparse.ArgumentParser(description='Ouster E2E model trainer.')
parser.add_argument('--train', type=str, help='folder containing the training data')
parser.add_argument('--test', type=str, help='folder containing the test data')
parser.add_argument('--only_curves', help='Whether to keep only curves and discard samples below curve_thresh', action='store_true')
parser.add_argument('--curve_thresh', type=float, help='the minimum absolute steering angle to be trained on [in radians] (default 0.0)', default=0.0)
parser.add_argument('--batch_size', type=int, help='the mini-batch size for training (default 32)', default=32)
parser.add_argument('--epoch_size', type=int, help='how many mini-batches per epoch (default 500)', default=500)
parser.add_argument('--epochs', type=int, help='how may epochs to train for (default 200)', default=200)
parser.add_argument('--lr', type=float, help='learning rate of optimizer (default 0.001)', default=0.001)
parser.add_argument('--dir', type=str, help='the directory to store tensorboard logs and saved models')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
print(current_time)
writer = SummaryWriter(log_dir=Path(args.dir) / 'runs' / current_time , max_queue=30, flush_secs=1)
print(device)

def yaw_aug(imgs, ys, ret_offsets=False):
    rots = torch.randint(-64, 64, (imgs.shape[0], )).to(device) # 22.5 degrees left and right (512 is center)
    center = rots + 512
    left = center - 256 + 1
    right = center + 256 + 1
    new_imgs = []
    for i in range(len(imgs)):
        new_imgs.append(imgs[i, :, :, left[i]:right[i]])
    offsets = (rots/1024.0 * 2 * np.pi).view(-1, 1)
    if ret_offsets:
        return torch.stack(new_imgs), ys + offsets, offsets
    return torch.stack(new_imgs), ys + offsets

def flip_aug(imgs, ys):
    flips = torch.randint(2, size=(imgs.shape[0], 1)).mul(2).sub(1).to(device)
    new_imgs = []
    for i in range(len(imgs)):
        if flips[i] == -1:
            new_img = imgs[i].flip(-1)
        else:
            new_img = imgs[i]
        new_imgs.append(new_img)
    return torch.stack(new_imgs), ys * flips

def evaluate_losses(loader, net):
    losses = torch.empty((0, 1)).to(device)
    with torch.no_grad():
        for batch_i, (imgs, steers, idx) in enumerate(loader):
            imgs, steers = imgs[:, :, :, 256:-256].to(device), steers.to(device)
            out = net(imgs)
            loss = torch.nn.L1Loss(reduction='none')(out, steers) / np.pi * 180
            losses = torch.cat([losses, loss])
            out, loss, residuals, imgs, steers = [None] * 5
            torch.cuda.empty_cache()
    return losses

def get_cl_weights(losses):
    weights = torch.nn.Softmax(dim=0)(losses).flatten().tolist()
    return torch.as_tensor(weights, dtype=torch.double)

dataset = OusterConcatDataset(args.train, curve_thresh=args.curve_thresh, only_curves=args.only_curves)
y_mean = dataset.get_steer_mean()
print(f'Train size: {len(dataset)}')
test_set = OusterConcatDataset(args.test, curve_thresh=args.curve_thresh, only_curves=False)
print(f'Test size: {len(test_set)}')
testloader = DataLoader(test_set, batch_size=100, num_workers=32, persistent_workers=True)

BATCH_SIZE = args.batch_size
weights = [1 for i in range(len(dataset))]
sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=BATCH_SIZE)
dataloader = DataLoader(dataset, persistent_workers=True, sampler=sampler, batch_size=BATCH_SIZE, drop_last=True, num_workers=32)

def train(models, trainloader, testloader, epochs=200, debug=False, loss_fn=torch.nn.MSELoss(), lr=0.001):
    nets = []
    opts = []
    scheds = []
    ls = [] # losses
    best = {
            'model': 'default',
            'train_loss': 'default',
            'maxoffset': 30,
            'epoch': 0,
            'MAE': 0}

    for model in models:
        torch.manual_seed(0)
        net = model()
        net.to(device)
        net = torch.nn.DataParallel(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=1e-05)

        net.eval()
        losses = evaluate_losses(testloader, net)
        net.train()

        nets.append(net)
        opts.append(optimizer)
        scheds.append(scheduler)
        ls.append(losses)
        
    train_step = 0
    for epoch in range(epochs):
        for net_i, net in enumerate(nets):
            for batch_i in range(args.epoch_size):#enumerate(trainloader):
                trainloader.sampler.weights = get_cl_weights(ls[net_i])
                data = next(iter(trainloader))
                inputs, labels, idx = data
                print(len(idx))
                inputs = inputs.to(device)
                #print(str(inputs.element_size() * inputs.nelement() / 1024 ** 2) + 'MiB')
                labels = labels.to(device)
                inputs, labels, offsets = yaw_aug(inputs, labels, True)
                #inputs, labels = flip_aug(inputs, labels)
                opts[net_i].zero_grad()
                torch.cuda.empty_cache()
                out = net(inputs)
                loss = torch.nn.L1Loss()(out, labels)
                loss_all = (out - labels).abs()
                ls[net_i][idx] = loss_all
                writer.add_scalars(f'Train/{type(loss_fn).__name__}', {type(net.module).__name__ : loss.item()}, train_step)
                loss.backward()
                opts[net_i].step()

                train_step += 1

        with torch.no_grad():

            for i, net in enumerate(nets):
                losses = torch.empty((0, 1)).to(device)
                steers_all = torch.empty((0, 1)).to(device)
                dummy_data = None
                for batch_i, data in enumerate(testloader):
                    #print(str(data[0].element_size() * data[0].nelement() / 1024 ** 2) + 'MiB')
                    imgs, steers = data[0][:, :, :, 256:-256].to(device), data[1].to(device)
                    dummy_data = imgs[:1]

                    net.eval()
                    out = net(imgs)
                    net.train()
                    loss = torch.nn.L1Loss(reduction='none')(out, steers) / np.pi * 180
                    losses = torch.cat([losses, loss])
                    steers_all = torch.cat([steers_all, steers])

                    out = None
                    loss = None
                    residuals = None
                    imgs, steers = None, None
                    torch.cuda.empty_cache()

                sorted_is = losses.argsort(0, True)[0]
                ls[net_i] = losses
                writer.add_scalars(f'Test/{type(loss_fn).__name__}_MAE', {type(net.module).__name__ : losses.mean()}, epoch)
                maxoffset = losses[sorted_is].item()
                writer.add_scalars(f'Test/{type(loss_fn).__name__}_maxoffset', {type(net.module).__name__ : maxoffset}, epoch)
                scheds[i].step(maxoffset)
                if i == 0:
                    losses = steers_all.abs() / np.pi * 180
                    sorted_is = losses.argsort(0, True)[0]
                    zero_maxoffset = losses[sorted_is].item()
                    writer.add_scalars(f'Test/{type(loss_fn).__name__}_MAE', {"Mean predictor": (steers_all - y_mean).abs().mean() / np.pi * 180,
                                                                                "Zero predictor": losses.mean()}, epoch)
                    mean_maxoffset = (steers_all - y_mean).abs().max() / np.pi * 180
                    writer.add_scalars(f'Test/{type(loss_fn).__name__}_maxoffset', {"Zero predictor" : zero_maxoffset,
                                                                                "Mean predictor": mean_maxoffset}, epoch)
                    
                if best['maxoffset'] > maxoffset:
                    best['model'] = type(net.module).__name__
                    best['train_loss'] = type(loss_fn).__name__
                    best['maxoffset'] = maxoffset
                    best['MAE'] = losses.mean()
                    best['epoch'] = epoch
                    net.module.eval()
                    torch.onnx.export(net.module, dummy_data, os.path.join(args.dir, f'{type(net.module).__name__}_{current_time}.onnx'), verbose=False)
                    net.module.train()
    writer.add_text(type(loss_fn).__name__, f"Best model: {best['model']}, epoch={best['epoch']}, MAE={best['MAE']}, maxoffset={best['maxoffset']}")

models =[OusterNetCropped] 
print('Training with MAE')
train(models, dataloader, testloader, epochs=args.epochs, loss_fn=torch.nn.L1Loss(), lr=args.lr)
#print('Training with MSE')
#train(models, dataloader, testloader, epochs=args.epochs, loss_fn=torch.nn.MSELoss(), lr=args.lr)
#print('Training with Huber loss')
#train(models, dataloader, testloader, epochs=args.epochs, loss_fn=torch.nn.SmoothL1Loss(), lr=args.lr)
