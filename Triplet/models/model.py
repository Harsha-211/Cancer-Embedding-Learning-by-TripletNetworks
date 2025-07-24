import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.Conv2d(128,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3),
            nn.ReLU(),
            nn.Conv2d(256,256,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*24*24,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )
    def forward_once(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    def forward(self,a,p,n):
        A = self.forward_once(a)
        P = self.forward_once(p)
        N = self.forward_once(n)
        return A,P,N
    