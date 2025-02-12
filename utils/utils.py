import ast
import numpy as np
import os 
import sys
import time
import torch

import pandas as pd
from PIL import Image
from scipy.special import logsumexp
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score
from scipy.special import logsumexp, expit

from models import get_model
from utils.crops import threshold_texture_crop
from utils.dataset import create_dataset

def ablate(ablation, method, parameter, processing_method, stride, window_size, metric, position, n, path, device, batch_size, num_workers):
    if method == 'DMID':
        progan_model, ldm_model = get_model(method, parameter, device)
        model = [progan_model, ldm_model]
    else:
        model = get_model(method, parameter, device)

    start_time = time.time()
    for name in os.listdir('datasets/'):
        print(name[:-4])
        dataloader = create_dataset(f'datasets/{name}', path, method, processing_method, batch_size, num_workers, stride, window_size, metric, position, n) 
        results = pd.DataFrame(columns=['src', 'label', 'logits'])

        df = pd.read_csv(f'datasets/{name}')
        results['src'] = df['filepath']
        results['label'] = df['label']
                
        y_scores = validate(dataloader, method, model, device, batch_size)
        results['logits'] = y_scores
        if ablation == 'stride':
            folder_path = f'./results/ablations/{ablation}/{method}_{parameter}/{str(stride)}' if parameter is not None else  f'./results/ablations/{ablation}/{method}/{str(stride)}'
        if ablation == 'window_size':
            folder_path = f'./results/ablations/{ablation}/{method}_{parameter}/{str(window_size)}' if parameter is not None else f'./results/ablations/{ablation}/{method}/{str(window_size)}'
        if ablation == 'metric':
            folder_path = f'./results/ablations/{ablation}/{method}_{parameter}/{metric}' if parameter is not None else f'./results/ablations/{ablation}/{method}/{metric}'
        if ablation == 'position':
            folder_path = f'./results/ablations/{ablation}/{method}_{parameter}/{position}' if parameter is not None else f'./results/ablations/{ablation}/{method}/{position}'
        if ablation == 'n':
            folder_path = f'./results/ablations/{ablation}/{method}_{parameter}/{n}' if parameter is not None else f'./results/ablations/{ablation}/{method}/{n}'
                                                                    
        os.makedirs(folder_path, exist_ok=True)
        results.to_csv(f'{folder_path}/{name}', index=False)

    end_time = time.time()  
    elapsed_time = end_time - start_time  

    time_df = pd.DataFrame({'Elapsed Time (seconds)': [elapsed_time]})
    time_df.to_csv(f'{folder_path}/time.csv', index=False)

    calculate_metrics(folder_path)
    print(f"Ablation analysis for the {ablation} parameter completed. Results saved in {folder_path}.") 


def evaluate(method, parameter, processing_method, path, device, batch_size, num_workers):
    if method == 'DMID':
        progan_model, ldm_model = get_model(method, parameter, device)
        model = [progan_model, ldm_model]
    else:
        model = get_model(method, parameter, device)

    start_time = time.time()
    for name in os.listdir('datasets/'):
        print(name[:-4])
        dataloader = create_dataset(f'datasets/{name}', path, method, processing_method, batch_size, num_workers)
        results = pd.DataFrame(columns=['src', 'label', 'logits'])

        df = pd.read_csv(f'datasets/{name}')
        results['src'] = df['filepath']
        results['label'] = df['label']
                
        y_scores = validate(dataloader, method, model, device, batch_size)
        results['logits'] = y_scores

        folder_path = f'./results/validations/{method}_{parameter}/{processing_method}/' if parameter is not None else f'./results/validations/{method}/{processing_method}/'
                                                    
        os.makedirs(folder_path, exist_ok=True)
        results.to_csv(f'{folder_path}/{name}', index=False)

    end_time = time.time()  
    elapsed_time = end_time - start_time  

    time_df = pd.DataFrame({'Elapsed Time (seconds)': [elapsed_time]})
    time_df.to_csv(f'{folder_path}/time.csv', index=False)

    calculate_metrics(folder_path)
    print(f"Evaluation completed. Results saved in {folder_path}.") 


def validate(dataloader, method, model, device, batch_size):
    outs = []

    iterator = tqdm(dataloader, desc="Processing Batches")
    for batch in iterator:
        images, labels = batch
        images = images.to(device)

        if method != 'PatchCraft':
            batch_size, num_crops, channels, height, width = images.shape
            images = images.view(batch_size * num_crops, channels, height, width)
        else: 
            batch_size, num_crops, num_patches, channels, height, width = images.shape
            images = images.view(batch_size * num_crops, num_patches, channels, height, width)

        if method != 'DMID':
            with torch.no_grad():
                logits = model.predict(images)  

            if isinstance(logits, list):
                logits = torch.stack([torch.tensor(logit) for logit in logits])

            logits = logits.view(batch_size, num_crops, -1)     
            outs.append(logits.mean(dim=1).cpu().numpy())  
        else:
            with torch.no_grad():
                progan_logits = model[0].predict(images)  
                ldm_logits = model[1].predict(images) 
            
            if isinstance(progan_logits, list):
                progan_logits = torch.stack([torch.tensor(logit) for logit in progan_logits])
            if isinstance(ldm_logits, list):
                ldm_logits = torch.stack([torch.tensor(logit) for logit in ldm_logits])

            progan_logits = progan_logits.view(batch_size, num_crops, -1)  
            ldm_logits = ldm_logits.view(batch_size, num_crops, -1)  
            aggregated_logits = torch.stack([torch.logsumexp(torch.stack([progan_logits[i], ldm_logits[i]]), 0) for i in range(batch_size)])
            outs.append(aggregated_logits.mean(dim=1).cpu())              

    outs = np.concatenate(outs, axis=0)

    return outs


def calculate_metrics(folder_path):
    metrics = []
    csv_files = [csv for csv in os.listdir(folder_path) if csv != 'time.csv' and csv != 'metrics.csv']
    for csv in csv_files:
        if csv == 'raise.csv': 
            continue
            
        elif csv in ['dalle2.csv', 'dalle3.csv', 'firefly.csv', 'stable-diffusion-2.csv', 'stable-diffusion-xl.csv']:
            df = pd.concat([pd.read_csv(os.path.join(folder_path, csv), converters={'logits': parse_list}), pd.read_csv(os.path.join(folder_path, 'raise.csv'), converters={'logits': parse_list})])
        else:
            df = pd.read_csv(os.path.join(folder_path, csv), converters={'logits': parse_list})
        
        y_true = df['label'].values
        y_logits = df['logits']
        y_pred = expit(y_logits) 

        accuracy = balanced_accuracy_score(y_true, y_pred > 0.5) * 100
        ap = average_precision_score(y_true, y_pred) * 100
        auc = roc_auc_score(y_true, y_pred) * 100
        metrics.append({'dataset': csv[:-4], 'acc': accuracy, 'ap': ap, 'auc': auc }) 

    df_results = pd.DataFrame(metrics)
    df_results = df_results.sort_values(by='dataset') 
    df_results.to_csv(os.path.join(folder_path, 'metrics.csv'), index=False)


def parse_list(x):
    return ast.literal_eval(x)


def threshold_texture_crop_ablation(method, parameter,  path, threshold, device):
    if method == 'DMID':
        progan_model, ldm_model = get_model(method, parameter, device)
        model = [progan_model, ldm_model]
    else:
        model = get_model(method, parameter, device)

    start_time = time.time()
    for name in os.listdir(f'datasets/'):
        print(name[:-4])
        df = pd.read_csv(f'./datasets/{name}')
        results = pd.DataFrame(columns=['src', 'label', 'logits'])
        results['src'] = df['filepath']
        results['label'] = df['label']

        output = []
        if method == 'PatchCraft':
            transform = None
        else:
            if method in ['UnivFD', 'RINE']:
                mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
            else:
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            transform = Compose([ToTensor(), Normalize(mean=mean, std=std)])
        
        for idx in tqdm(range(len(df)), desc=f"Processing Images in {name}"):
            outs = []
            label = df.iloc[idx, 1]    
            img_path = os.path.join(path, df.iloc[idx, 0])

            image = Image.open(img_path).convert('RGB')
            cropped_images = threshold_texture_crop(image=image, threshold=threshold)
            for crop in cropped_images:
                if method == 'PatchCraft':
                    crop = rptc_processing(crop).unsqueeze(0).to(device) 
                else:
                    crop = transform(crop).unsqueeze(0).to(device)
                    
                if method != 'DMID':
                    out = model.predict(crop)
                else:
                    with torch.no_grad():
                        progan_logit = model[0].predict(crop)  
                        ldm_logit = model[1].predict(crop) 
                        out = torch.logsumexp(torch.stack([torch.tensor(progan_logit), torch.tensor(ldm_logit)]), dim=0)

                outs.append(out)
            output.append(outs)

        results['logits'] = output
        
        folder_path = f'./results/ablations/threshold_texture_crop/{method}_{parameter}/{str(threshold)}' if parameter is not None else  f'./results/ablations/threshold_texture_crop/{method}/{str(threshold)}'    
        os.makedirs(folder_path, exist_ok=True)
        results.to_csv(f'{folder_path}/{name}', index=False)

    end_time = time.time()  
    elapsed_time = end_time - start_time  

    time_df = pd.DataFrame({'Elapsed Time (seconds)': [elapsed_time]})
    time_df.to_csv(f'{folder_path}/time.csv', index=False)

    calculate_metrics(folder_path)
    print(f"Threshold TextureCrop ablation completed. Results saved in {folder_path}.") 
