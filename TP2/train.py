import torch.nn as nn
import torch.optim as optim

import torch
from torch.utils.data import DataLoader, random_split
# (Importez votre CardioDataset et vos loaders ici)
from dataset import CardioDataset



def get_loaders(csv_path="data/cardio_train.csv", batch_size=64, seed=42):
    dataset = CardioDataset(csv_path)
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = get_loaders()
batch = next(iter(train_loader))



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() # Sortie binaire [0, 1]
        )

    def forward(self, x):
        return self.net(x)

# Initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size=batch['features'].shape[1], hidden_size=128).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

l1_lambda = 0.1
l2_lambda = 0

for epoch in range(10):
    model.train()
    for batch in train_loader:
        inputs, targets = batch["features"].to(device), batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        base_loss = criterion(outputs, targets)
        
        # Calcul de la pénalité L1 (somme des valeurs absolues des poids)
        l1_penalty = sum(p.abs().sum() for p in model.parameters())
        
        # Calcul de la pénalité L2 (somme des carrés des poids)
        l2_penalty = sum((p ** 2).sum() for p in model.parameters())
        
        # Loss totale
        loss = base_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty
        
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch+1}: loss = {loss.item():.4f}")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in val_loader:
        inputs, targets = batch["features"].to(device), batch["labels"].to(device)
        preds = (model(inputs) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.numel()
print(f"val acc = {correct/total:.4f}")