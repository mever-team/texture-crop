import argparse
import torch
from utils.utils import ablate, threshold_texture_crop_ablation

torch.backends.cudnn.deterministic=True 
torch.backends.cudnn.benchmark=False


def ablation_study(ablation, method, parameter, path, device, batch_size, num_workers):
    if ablation == 'stride':
        for stride in [112, 336]:
            print('\nMethod: ', method, 'Parameter: ', parameter) if parameter is not None else print('\nMethod: ', method)
            print('Ablation: Stride')
            print('Stride: ', stride)
            ablate(ablation=ablation, method=method, parameter=parameter,  processing_method='texture_crop', stride=stride, window_size=224, metric='ghe', position='top', n=10, path=path, device=device, batch_size=batch_size, num_workers=num_workers)
    elif ablation == 'window_size':
        for stride in [512, 1024]:
            print('\nMethod: ', method, 'Parameter: ', parameter) if parameter is not None else print('\nMethod: ', method)
            print('Ablation: Window size')
            print('Window size: ', window_size)
            ablate(ablation=ablation, method=method, parameter=parameter,  processing_method='texture_crop', stride=224, window_size=window_size, metric='ghe', position='top', n=10, path=path, device=device, batch_size=batch_size, num_workers=num_workers)
    elif ablation == 'metric':
        for metric in ['ac', 'td']:
            print('\nMethod: ', method, 'Parameter: ', parameter) if parameter is not None else print('\nMethod: ', method)
            print('Ablation: Metric')
            print('Metric: ', metric)
            ablate(ablation=ablation, method=method, parameter=parameter, processing_method='texture_crop', stride=224, window_size=224, metric=metric, position='top', n=10, path=path, device=device, batch_size=batch_size, num_workers=num_workers)
    elif ablation == 'n':
        for n in [1, 5]:
            print('\nMethod: ', method, 'Parameter: ', parameter) if parameter is not None else print('\nMethod: ', method)
            print('Ablation: N')
            print('N: ', n)
            ablate(ablation=ablation, method=method, parameter=parameter,  processing_method='texture_crop', stride=224, window_size=224, metric='ghe', position='top', n=n, path=path, device=device, batch_size=batch_size, num_workers=num_workers)
    elif ablation == 'position':
        for position in ['bottom']:
            print('\nMethod: ', method, 'Parameter: ', parameter) if parameter is not None else print('\nMethod: ', method)
            print('Ablation: Position')
            print('Position: ', position)
            ablate(ablation=ablation, method=method, parameter=parameter, processing_method='texture_crop', stride=224, window_size=224, metric='ghe', position=position, n=10, path=path, device=device, batch_size=batch_size, num_workers=num_workers)
    elif ablation == 'threshold_texture_crop':
            print('\nMethod: ', method, 'Parameter: ', parameter) if parameter is not None else print('\nMethod: ', method)
            print('Ablation: Threshold Texture Crop')
            threshold_texture_crop_ablation(method=method, parameter=parameter, path=path, threshold=5, device=device)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ablation', '-a', type=str, default=None, help='Ablation study to run: stride, window_size, metric, n, p or threshold_texture_crop.')
    parser.add_argument('--method', '-m', type=str, help='GramNet, CNNDetect, GANID, DIMD, UnivFD, RINE, PatchCraft')
    parser.add_argument('--parameter', '-par', type=str, default=None, help='0.1, 0.5 for CNNDetection | progan, stylegan2 for GANID | 4, ldm for RINE')
    parser.add_argument('--path', '-pt', type=str, default='/mnt/cephfs/home/dekonstantinidou/datasets', help='Path to the folders containing the images.')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='Device to run the experiments on.')
    parser.add_argument('--batch_size', '-bs', type=int, default=16, help='Batch size for processing.')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers.')

    args=parser.parse_args()

    if args.ablation=='threshold_texture_crop':
        args.batch_size=1

    ablation_study(ablation=args.ablation, method=args.method, parameter=args.parameter, path=args.path, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers)