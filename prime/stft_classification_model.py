
import torch
from torch import nn, einsum
import torch.nn.functional as F

import torch.nn.functional as F 
class STFTClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resample = nn.Upsample(size=(32, 32))
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            
        )
        self.out = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128*2*2, 1024), 
            nn.ReLU(), 
            nn.Linear(1024, 4), 
        )
    def forward(self, x):
        N, C, T = x.shape 
        x = x[:, 2, :]
        y = torch.stft(x, 256, hop_length=128, return_complex=True)
        y = torch.abs(y)
        y /= torch.max(y) + 1e-6
        y = y.unsqueeze(1)
        y = self.resample(y)
        y = self.classifier(y) 
        #print(y.shape)
        y = self.out(y)

        return y 


class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()
    def forward(self, x, d):
        self.ce(x, d)
        
if __name__ == "__main__":
    model = STFTModel()
    x = torch.ones([32, 3, 10240]) 
    y1, y2, y3  = model(x)
    print(y1.shape, y2.shape, y3.shape)
