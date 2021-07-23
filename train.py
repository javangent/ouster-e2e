import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data import OusterConcatDataset
from torch.utils.tensorboard import SummaryWriter
from model import *
import pandas as pd
from torchvision import models, io
import os
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import time 

def get_args():
    parser = argparse.ArgumentParser(description='Ouster E2E model trainer.')
    parser.add_argument('--train', type=str, help='folder containing the training data')
    parser.add_argument('--test', type=str, help='folder containing the test data')
    parser.add_argument('--q2', type=float, help='The amount represenation of curves in batches (default 0.5)', default=0.5)
    parser.add_argument('--bins', type=int, help='Number of bins to use to create differen steering classes for sample weight calculation', default=40)
    parser.add_argument('--bw', type=float, help='KDE kernel size to be used for pdf estimation', default=None)
    parser.add_argument('--only_curves', help='Whether to keep only curves and discard samples below curve_thresh', action='store_true')
    parser.add_argument('--curve_thresh', type=float, help='threshold to distinguish between curves and non-curves [in radians] (default 0.005)', default=0.005)
    parser.add_argument('--batch_size', type=int, help='the mini-batch size for training (default 128)', default=128)
    parser.add_argument('--epoch_size', type=int, help='how many mini-batches per epoch (default 500)', default=500)
    parser.add_argument('--epochs', type=int, help='how may epochs to train for (default 50)', default=50)
    parser.add_argument('--lr', type=float, help='learning rate of optimizer (default 0.02)', default=0.02)
    parser.add_argument('--dir', type=str, help='the directory to store tensorboard logs and saved models')
    parser.add_argument('--gpus', type=int, help='the number of GPUs to use', default=2)
    parser.add_argument('--num_steers', type=int, help='the number of steering angles to predict', default=1)
    parser.add_argument('--use_diff', help='use difference between current and last frame as input image', action='store_true')
    parser.add_argument('--image_format', type=str, help='format of input image (default "ria" - range, intensity, ambience)', default="ria")
    parser.add_argument('--comment', type=str, help='comment used as a prefix to run folder', default="")
    args = parser.parse_args()

    args.world_size = args.gpus
    return args


def rad_to_pix(rad):
    return rad / np.pi * 512

def pix_to_rad(pix):
    return pix / 512 * np.pi

def yaw_aug(imgs, ys, device, std, ret_offsets=False):
    #rots = torch.randint(-32, 32, ys.shape, device=device) # 11.25 degrees left and right (512 is center)
    #offsets = pix_to_rad(rots)
    offsets = torch.normal(0, 2*std, (len(ys), 1), device=device)
    rots = rad_to_pix(offsets).int()
    center = rots + 512
    left = center - 256
    right = center + 256
    new_imgs = []
    ys[:, [0]] += offsets # only for first angle
    for i in range(len(imgs)):
        new_imgs.append(imgs[i, :, :, left[i]:right[i]])
    if ret_offsets:
        return torch.stack(new_imgs), ys, offsets
    return torch.stack(new_imgs), ys

def flip_aug(imgs, ys, device):
    flips = torch.randint(2, size=(imgs.shape[0], 1)).mul(2).sub(1).to(device)
    new_imgs = []
    for i in range(len(imgs)):
        if flips[i] == -1:
            new_img = imgs[i].flip(-1)
        else:
            new_img = imgs[i]
        new_imgs.append(new_img)
    return torch.stack(new_imgs), ys * flips

def evaluate(rank, net, testloader, y_mean):
    net.eval()
    out_all = []
    steers_all = []
    dummy_data = None
    imgs = None
    for batch_i, data in enumerate(testloader):
        imgs = data["image"][:, :, :, 256:-256].to(rank, non_blocking=True)
        steers = data["steer"].to(rank, non_blocking=True)
        if batch_i == 0:
            dummy_data = imgs[:1]
        out = net(imgs)
        del imgs
        out_all.append(out)
        del out
        steers_all.append(steers)
        del steers
    net.train()

    out_all = torch.cat(out_all)
    steers_all = torch.cat(steers_all)

    zero_predictor_losses = steers_all.abs() / np.pi * 180
    zero_predictor_loss_sum = zero_predictor_losses.sum()
    maxoffset_zero = zero_predictor_losses.max()
    del zero_predictor_losses

    mean_predictor_losses = (y_mean - steers_all).abs() / np.pi * 180
    mean_predictor_loss_sum = mean_predictor_losses.sum()
    maxoffset_mean = mean_predictor_losses.max()
    del mean_predictor_losses

    net_losses = (out_all - steers_all).abs() / np.pi * 180
    net_loss_sum = net_losses.sum() 
    maxoffset_net = net_losses.max()
    del net_losses

    del out_all
    del steers_all

    loss_sums = [net_loss_sum, zero_predictor_loss_sum, mean_predictor_loss_sum]
    maxoffsets = [maxoffset_net, maxoffset_zero, maxoffset_mean]
    return loss_sums, maxoffsets, dummy_data

def init_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
    	backend='nccl',
   		init_method='env://',
    	world_size=world_size,
    	rank=rank
    )   


def train(rank, args):
    init_process(rank, args.gpus)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    out_dir = Path(args.dir) / 'runs' / (current_time + "_" + args.comment)
    if rank == 0:
        print(current_time)
        writer = SummaryWriter(log_dir=out_dir , max_queue=30, flush_secs=1)
    models = [OusterNet]
    loss_fn=torch.nn.L1Loss()

    frame_dist = 1 if args.use_diff else 0
    dataset = OusterConcatDataset(args.train, curve_thresh=args.curve_thresh, only_curves=args.only_curves, frame_dist=frame_dist, future_steer_dist=10, num_steers=args.num_steers, image_format=args.image_format)
    y_mean = dataset.get_steer_mean()
    std = dataset.get_std()
    if rank == 0:
        print(f'Train size: {len(dataset)}')
    test_set = OusterConcatDataset(args.test, curve_thresh=args.curve_thresh, only_curves=False, frame_dist=frame_dist, future_steer_dist=10, num_steers=args.num_steers, image_format=args.image_format)
    if rank == 0:
        print(f'Test size: {len(test_set)}')
    dist_sampler = DistributedSampler(test_set, shuffle=False)
    testloader = DataLoader(test_set, batch_size=200//args.gpus, num_workers=16//args.gpus, sampler=dist_sampler, persistent_workers=True, pin_memory=True)

    BATCH_SIZE = args.batch_size // args.gpus

    weights = dataset.get_threshold_weights(0.5)

    args.epoch_size = len(dataset)//BATCH_SIZE
    sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=args.epoch_size * BATCH_SIZE)
    trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, persistent_workers=True, sampler=sampler, drop_last=True, num_workers=16//args.gpus, pin_memory=True)
    nets = []
    opts = []
    scheds = []
    best = {
            'model': 'default',
            'train_loss': 'default',
            'maxoffset': 100,
            'epoch': 0,
            'MAE': 0}

    for model in models:
        torch.manual_seed(rank)
        net = model(num_steers=args.num_steers, input_shape=dataset[0]["image"].shape)
        net.to(rank)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        #optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.6, verbose=True)

        nets.append(net)
        opts.append(optimizer)
        scheds.append(scheduler)
        
    train_step = 0
    for epoch in range(args.epochs):
        ### TRAIN FOR SINGLE EPOCH ###
        for batch_i, data in enumerate(trainloader):
            inputs, labels = data["image"], data["steer"]
            inputs = inputs.to(rank, non_blocking=True)
            #print(str(inputs.element_size() * inputs.nelement() / 1024 ** 2) + 'MiB')
            labels = labels.to(rank, non_blocking=True)
            inputs, labels, offsets = yaw_aug(inputs, labels, rank, std, True)
            for net_i, net in enumerate(nets):
                opts[net_i].zero_grad()
                loss = loss_fn(net(inputs), labels)
                if rank == 0 and train_step % 500 == 0:
                    writer.add_scalars(f'Train/{type(loss_fn).__name__}', {type(net.module).__name__ : loss.item()}, train_step)
                loss.backward()
                opts[net_i].step()
            train_step += 1

        
        with torch.no_grad():
            data_len = len(test_set)
            for i, net in enumerate(nets):
                ### START EVALUATION ###
                loss_sums, maxoffsets, dummy_data = evaluate(rank, net, testloader, y_mean)
                ### END EVALUATION ###

                ### REDUCE RESULTS FROM OTHER GPUS ###
                losses = [0, 0, 0]
                for j in range(3): # net, zero, train mean
                    dist.all_reduce(loss_sums[j], op=dist.ReduceOp.SUM)
                    losses[j] = loss_sums[j] / data_len
                    dist.all_reduce(maxoffsets[j], op=dist.ReduceOp.MAX)
                ### END REDUCING ###

                scheds[i].step(maxoffsets[0]) # Update LR scheduler

                if rank == 0:
                    ### BEGIN LOGGING RESULTS ###
                    writer.add_scalars(
                            f'Test/{type(loss_fn).__name__}_MAE',
                            {type(net.module).__name__ : losses[0]},
                            epoch)
                    writer.add_scalars(f'Test/{type(loss_fn).__name__}_maxoffset',
                            {type(net.module).__name__ : maxoffsets[0]},
                            epoch)
                    if i == 0:
                        writer.add_scalars(f'Test/{type(loss_fn).__name__}_MAE',
                                {
                                    "Mean predictor": losses[2],
                                    "Zero predictor": losses[1]
                                },
                                epoch)
                        writer.add_scalars(
                                f'Test/{type(loss_fn).__name__}_maxoffset',
                                {
                                    "Zero predictor" : maxoffsets[1],
                                    "Mean predictor": maxoffsets[2]
                                },
                                epoch)
                    ### END LOGGING RESULTS ###

                    ### SAVE BEST MODEL ###
                    if best['maxoffset'] > maxoffsets[0].item():
                        best['model'] = type(net.module).__name__
                        best['train_loss'] = type(loss_fn).__name__
                        best['maxoffset'] = maxoffsets[0].item()
                        best['MAE'] = losses[0].item()
                        best['epoch'] = epoch
                        net.module.eval()
                        torch.save(net.module.state_dict(), out_dir / f'{type(net.module).__name__}_{current_time}.pt')
                        torch.onnx.export(net.module, dummy_data, out_dir / f'{type(net.module).__name__}_{current_time}.onnx', verbose=False)
                        traced_module = torch.jit.trace(net.module, dummy_data)
                        traced_module.save(out_dir / f'{type(net.module).__name__}_{current_time}.ts')
                        net.module.train()
        torch.distributed.barrier()
    if rank == 0:
        writer.add_text(type(loss_fn).__name__, f"Best model: {best['model']}, epoch={best['epoch']}, MAE={best['MAE']}, maxoffset={best['maxoffset']}")
    dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args()
    mp.spawn(train, nprocs=args.gpus, args=(args, ), join=True)

