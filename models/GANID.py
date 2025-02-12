import torch.nn as nn 
import torch

from models.networks.resnet50nodown import resnet50nodown
import numpy as np

class GANID(nn.Module):

    def __init__(self):
        super(GANID, self).__init__()
        self.model = resnet50nodown(num_classes=1)

    def forward(self, x):
        return self.model(x)

    def load_weights(self, training):
        state_dict = torch.load(f'models/weights/GANID/gandetection_resnet50nodown_{training}.pth', map_location='cpu')
        try:
            self.model.load_state_dict(state_dict['model'])
            if ('module._conv_stem.weight' in state_dict['model']) or ('module.fc.fc1.weight' in state_dict['model']) or ('module.fc.weight' in state_dict['model']):
                self.load_state_dict({key[7:]: state_dict['model'][key] for key in state_dict['model']})
            else:
                self.model.load_state_dict(state_dict['model'])

        except:
            print('Loading state dict failed. Trying to load without model prefix.')
            self.model.load_state_dict(state_dict)

    def predict(self, img):
        with torch.no_grad():
            logits = self.forward(img)
        
        return logits.flatten().tolist()
