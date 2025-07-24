import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from models.model import TripletNet
from utils.data_util import TripletFlatDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
dataset = TripletFlatDataset(root_dir = "D:/Pythonn/Cancer/Data", transform=transform)
loader = DataLoader(dataset,batch_size=32,shuffle=True)

model = TripletNet().to(device)
criterion = nn.TripletMarginLoss(margin=1.0,p=2)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
torch.cuda.empty_cache()

for epoch in range(10):
    print(f"Starting epoch {epoch+1}")
    total_loss = 0

    
    for a, p, n, label in loader:
        a, p, n = a.to(device), p.to(device), n.to(device)

        A, P, N = model(a, p, n)
        loss = criterion(A, P, N)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")