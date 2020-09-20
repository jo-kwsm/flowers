import glob
import os.path as osp
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as pyplot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

import loader
from train import train_model

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

data_path = "./data"
data_path_list=loader.make_datapath_list(data_path)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = loader.ImageTransform(resize=resize, mean=mean, std=std)
print(len(data_path_list))

batch_size = 32
epochs = 50


dataset=loader.FlowersDataset(data_path_list,transform=transform)
train_size = int(len(dataset)*0.8)
val_size = len(dataset)-train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict={"train":train_dataloader, "val":val_dataloader}

net = models.resnet50(num_classes = 5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
train_model(net, dataloaders_dict, criterion, optimizer, epochs)
