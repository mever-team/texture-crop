import os
import csv
import torch
from PIL import Image
from scipy.special import expit
from scipy.special import logsumexp
import argparse
import numpy as np
from tqdm import tqdm 
from torchvision.transforms import Compose, Normalize, ToTensor

from models import get_model
from utils.dataset import crop


def process(image, method):
    if method == 'PatchCraft':
        transform = None
    else:
        if method in ['UnivFD', 'RINE']:
            mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose([ToTensor(), Normalize(mean=mean, std=std)])
    
        if method != 'PatchCraft':
            if isinstance(image, list):    
                cropped_images = [transform(crop) for crop in image]
                image_tensor = torch.stack(cropped_images) 
            else:
                image_tensor = transform(image)
        else:
            if isinstance(image, list):    
                cropped_images = [rptc_processing(crop) for crop in cropped_images]
                image_tensor = torch.stack(cropped_images) 
            else: 
                image_tensor = rptc_processing(image)
    return image_tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--method', '-m', type=str, default='RINE', help='GramNet, CNNDetect, GANID, DMID, UnivFD, RINE, PatchCraft')
    parser.add_argument('--parameter', '-p', type=str, default=None, help='0.1 or 0.5 for CNNDetect method / progan or stylegan2 for GANID method / 4 or ldm for Rine method')
    parser.add_argument('--image_path', '-i', type=str, help='Path to the image')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device to run experiments on')
    args = parser.parse_args()

    device = args.device

    print(f"Results for image {args.image_path}:")

    image = Image.open(args.image_path).convert('RGB')
    for processing_method in ['resize', 'centercrop', 'tencrop', 'texture_crop', 'threshold_texture_crop']:
        crops = crop(image, processing_method)

        if args.method == 'DMID':
            progan_model, ldm_model = get_model(args.method, args.parameter, args.device)
            images = process(crops, 'DMID')
            images.to(args.device)

            progan_logits = progan_model.predict(images)
            ldm_logits = ldm_model.predict(images)

            logits = []
            for i, logit in enumerate(progan_logits):
                logits.append(logsumexp([np.array(logit), np.array(ldm_logits[i])], axis=0))
        else:
            model = get_model(args.method, args.parameter, args.device)
            images = process(crops, args.method)
            images.to(args.device)
            logits = model.predict(images)

        avg = sum(logits) / len(logits)
        output = expit(avg)


        print(f"{processing_method}: {output}")
        