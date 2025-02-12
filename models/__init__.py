from models.CNNDetect import CNNDetect
from models.UnivFD import UnivFD
from models.GANID import GANID
from models.GramNet import GramNet
from models.Rine import RineModel
from models.DMID import DMID
from models.PatchCraft import Net as PatchCraftNet

import re
import torch

VALID_METHODS = ['CNNDetect', 'DMID', 'GANID', 'GramNet', 'RINE', 'UnivFD', 'PatchCraft']

def get_model(method, parameter, device):
    assert method in VALID_METHODS

    if method == 'DMID':
        progan_model = DMID()
        progan_model.load_weights(ckpt = 'models/weights/DMID/Grag2021_progan/model_epoch_best.pth')
        progan_model.eval()
        progan_model = progan_model.to(device)
        
        ldm_model = DMID()
        ldm_model.load_weights(ckpt = 'models/weights/DMID/Grag2021_latent/model_epoch_best.pth')
        ldm_model.eval()
        ldm_model = ldm_model.to(device)

        return progan_model, ldm_model

    else:
        if method == 'CNNDetect':
            model = CNNDetect() 
            if parameter == '0.1' or parameter == '0.5':
                model.load_weights(prob = parameter)
            else:
                raise ValueError(f"Unsupported parameter value: {parameter}. Supported values are '0.1' and '0.5'.")

        elif method == 'GANID':
            model = GANID()
            if parameter == 'progan' or parameter == 'stylegan2':
                model.load_weights(training = parameter)
            else:
                raise ValueError(f"Unsupported parameter value: {parameter}. Supported values are 'progan' and 'stylegan2'.")

        elif method == 'GramNet':
            model = GramNet()
            model.load_weights()


        elif method == 'RINE':
            if parameter == '1' or parameter == '2' or parameter == '4' or parameter == 'ldm':
                model = RineModel(ncls = parameter)
                model.load_weights(ncls = parameter)
            else:
                raise ValueError(f"Unsupported parameter value: {parameter}. Supported values are '1', '2', '4' and 'ldm'.")

        elif method == 'UnivFD':
            model = UnivFD()
            model.load_weights()
        
        elif method == 'PatchCraft':
            model = PatchCraftNet()
            model.load_weights()
        
        model.eval()
        model = model.to(device)

        return model