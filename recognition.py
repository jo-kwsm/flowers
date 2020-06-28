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

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

data_path = "./data"
data_path_list=loader.make_datapath_list(data_path)
dataset=loader.FlowersDataset(data_path_list)