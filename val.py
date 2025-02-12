import argparse
import numpy as np
import os 
import pandas as pd
import sys
import time
import torch

from PIL import Image
from scipy.special import logsumexp, expit
from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score
from tqdm import tqdm

from utils.utils import evaluate

torch.backends.cudnn.deterministic=True 
torch.backends.cudnn.benchmark=False
    

if __name__ =='__main__':
    start_time=time.time()
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--method', '-m', type=str, help='GramNet, CNNDetect, GANID, DIMD, UnivFD, RINE, PatchCraft')
    parser.add_argument('--parameter', '-par', type=str, default=None, help='0.1, 0.5 for CNNDetection | progan, stylegan2 for GANID | 4, ldm for RINE')
    parser.add_argument('--processing_method', '-pm', type=str, default='texture_crop', help='resize, centercrop, tencrop, texture_crop')
    parser.add_argument('--path', '-pt', type=str, default='/mnt/cephfs/home/dekonstantinidou/datasets', help='The path to the folders containing the images')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='Device to run the experiments on')
    parser.add_argument('--batch_size', '-bs', default=128, type=int, help='Batch size')
    parser.add_argument('--num_workers', '-nw', type=int, default=12, help='Number of workers')
    args=parser.parse_args()

    print("\nMethod: ", args.method)
    print("Processing method: ", args.processing_method)
    evaluate(method=args.method, parameter=args.parameter, processing_method=args.processing_method, path=args.path, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers)