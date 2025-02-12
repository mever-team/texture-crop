import copy
import cv2
import os

from scipy import fftpack
from io import BytesIO 
from PIL import Image 
import pandas as pd
import numpy as np 

import torch 
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Resize, RandomCrop, TenCrop, Lambda
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.crops import *


def crop(image, processing_method, stride=224, window_size=224, metric='ghe', position='top', n=10):
    if processing_method == 'resize':
        transform = Resize((224,224))
        cropped_images = [transform(image)]
    elif processing_method == 'centercrop':
        transform = CenterCrop(224)
        cropped_images = [transform(image)]
    elif processing_method == 'tencrop':
        transform = TenCrop(224)
        cropped_images = list(transform(image))
    elif processing_method == 'texture_crop':
        cropped_images = texture_crop(image, stride, window_size, metric, position, n)
    elif processing_method == 'threshold_texture_crop':
        cropped_images = threshold_texture_crop(image)
    return cropped_images


class ImageDataset(Dataset):
    def __init__(self, csv_file, path, method, processing_method, stride, window_size, metric, position, n, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.path = path  
        self.method = method  
        self.processing_method = processing_method
        self.stride = stride
        self.window_size = window_size  
        self.metric = metric  
        self.position = position
        self.n = n
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = self.data_frame.iloc[idx, 0]  
        img_path = os.path.join(self.path, img_filename)

        label = self.data_frame.iloc[idx, 1]    

        image = Image.open(img_path).convert('RGB')
        cropped_images = crop(image, self.processing_method, self.stride, self.window_size, self.metric, self.position, self.n) 

        if self.method != 'PatchCraft':
            if isinstance(cropped_images, list):    
                cropped_images = [self.transform(crop) for crop in cropped_images]
                image_tensor = torch.stack(cropped_images) 
            else:
                image_tensor = self.transform(cropped_images)
        else:
            if isinstance(cropped_images, list):    
                cropped_images = [rptc_processing(crop) for crop in cropped_images]
                image_tensor = torch.stack(cropped_images) 
            else: 
                image_tensor = rptc_processing(cropped_images)

        return image_tensor, label



def create_dataset(csv_file, path, method, processing_method, batch_size, num_workers, stride=224, window_size=224, metric='ghe', position='top', n=10, shuffle=False):
    if method == 'PatchCraft':
        transform = None
    else:
        if method in ['UnivFD', 'RINE']:
            mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose([ToTensor(), Normalize(mean=mean, std=std)])

    dataset = ImageDataset(csv_file, path, method, processing_method, stride, window_size, metric, position, n, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
    return dataloader


def ED(img):
    r1, r2 = img[:, 0:-1, :], img[:, 1::, :]
    r3, r4 = img[:, :, 0:-1], img[:, :, 1::]
    r5, r6 = img[:, 0:-1, 0:-1], img[:, 1::, 1::]
    r7, r8 = img[:, 0:-1, 1::], img[:, 1::, 0:-1]
    s1 = torch.sum(torch.abs(r1 - r2)).item()
    s2 = torch.sum(torch.abs(r3 - r4)).item()
    s3 = torch.sum(torch.abs(r5 - r6)).item()
    s4 = torch.sum(torch.abs(r7 - r8)).item() 

    return s1 + s2 + s3 + s4


def rptc_processing(img, patchNum = 3):
    num_block = int(pow(2, patchNum))
    patchsize = int(224 / num_block)
    randomcrop = RandomCrop(patchsize)
    
    minsize = min(img.size)
    if minsize < patchsize:
        img = Resize((patchsize,patchsize))(img)
    
    img = ToTensor()(img)

    imgori = img.clone().unsqueeze(0)
    img_template = torch.zeros(3, 224, 224)
    img_crops = []
    for i in range(num_block * num_block * 3):
        cropped_img = randomcrop(img)
        texture_rich = ED(cropped_img)
        img_crops.append([cropped_img, texture_rich])

    img_crops = sorted(img_crops, key=lambda x:x[1])

    count = 0
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:,ii*patchsize:(ii+1)*patchsize,jj*patchsize:(jj+1)*patchsize] = img_crops[count][0]
            count += 1
    img_poor = img_template.clone().unsqueeze(0)

    count = -1
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:,ii*patchsize:(ii+1)*patchsize,jj*patchsize:(jj+1)*patchsize] = img_crops[count][0]
            count -= 1
    img_rich = img_template.clone().unsqueeze(0)
    img = torch.cat((img_poor,img_rich),0)
    
    return img