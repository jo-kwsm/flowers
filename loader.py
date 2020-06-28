import glob
import os.path as osp
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

class ImageTransform():
  def __init__(self,resize,mean,std):
    self.data_transform = {
      "train" : transforms.Compose([
        transforms.RandomResizedCrop(resize,scalse=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
      ]),
      "val" : transforms.Compose([
        transforms.Resized(resize,scalse=(0.5,1.0)),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
      ])
    }
  
  def __call__(self, img, phase="train"):
    return self.data_transform[phase](img)


def make_datapath_list(rootpath):
  target_path = osp.join(rootpath+"/**/*.jpg")
  print(target_path)

  path_list=[]

  for path in glob.glob(target_path):
    path_list.append(path)

  return path_list

class FlowersDataset(data.Dataset):
  def __init__(self, file_list, transform=None, phase="train"):
    self.file_list = file_list
    self.transform = transform
    self.phase = phase

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, index):
    img_path = self.file_list[index]
    img = Image.open(img_path)
    img_transformed = self.transform(img, self.phase)
    label=img_path.split("/")[2]

    label_change = {"daisy":0, "dandelion":1, "rose":2, "sunflower":3, "tulip":4}
    label = label_change[label]

    return img_transformed, label