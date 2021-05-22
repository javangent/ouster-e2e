import torch
import torchvision.transforms.functional as TF
from torchvision import models


class CommaNet(torch.nn.Module):
    def __init__(self):
        super(type(self), self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 8, 4),
            torch.nn.ELU(),
            torch.nn.Conv2d(16, 32, 5, 2),
            torch.nn.ELU(),
            torch.nn.Conv2d(32, 64, 5, 2),
            torch.nn.Flatten(),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.5),
            #torch.nn.Linear(3648, 100),
            #torch.nn.ELU(),
            #torch.nn.Linear(100, 50),
            #torch.nn.ELU(),
            #torch.nn.Linear(50, 10),
            #torch.nn.ELU(),
            #torch.nn.Linear(10, 1),
        )
    def forward(self, x):
        return self.net(x)

class Crop(torch.nn.Module):
    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):
        return img[..., self.top:self.top + self.height, self.left:self.left + self.width]
        #return TF.crop(img, self.top, self.left, self.height, self.width)


road_crop = Crop(62, 0, 128-62 - 5, 512)

class PilotNet(torch.nn.Module):
    def __init__(self):
        super(type(self), self).__init__()
        self.net = torch.nn.Sequential(
            road_crop,
            torch.nn.Conv2d(3, 24, 5, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 36, 5, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(36, 48, 5, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(3648, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
    def forward(self, x):
        return self.net(x)

class OusterNet(torch.nn.Module):
    def __init__(self, num_steers=1, fov=256, input_shape=(3, 128, 512)):
        super(type(self), self).__init__()
        self.features = torch.nn.Sequential(
            Crop(62, 256-fov//2, 128-62 - 5, fov),
            torch.nn.Conv2d(input_shape[0], 24, 5, 2),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 36, 5, 2),
            torch.nn.BatchNorm2d(36),
            torch.nn.ReLU(),
            torch.nn.Conv2d(36, 48, 5, 2),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 64, 3, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        dummy = torch.randn((2, *input_shape))
        flatten_len = self.features(dummy).shape[1]
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(flatten_len, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.BatchNorm1d(50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, num_steers),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

