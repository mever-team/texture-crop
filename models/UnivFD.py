
from models.networks.clip import clip 

import torch.nn as nn 
import torch


class UnivFD(nn.Module):
    def __init__(self, num_classes=1):
        super(UnivFD, self).__init__()
        self.model, _ = clip.load("ViT-L/14", device="cpu")
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.model.encode_image(x) 
        return self.fc(features)
    
    def load_weights(self):
        state_dict = torch.load('models/weights/UnivFD/fc_weights.pth', map_location='cpu')
        self.fc.load_state_dict(state_dict)

    def predict(self, img):
        with torch.no_grad():
            logits = self.forward(img)
            return logits.flatten().tolist()
