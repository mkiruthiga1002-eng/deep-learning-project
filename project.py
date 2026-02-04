import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])

full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
full_test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_data = [(x, y) for x, y in full_train if y == 1]

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(full_test, batch_size=128, shuffle=False)

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 400)
        self.fc3 = nn.Linear(400, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon, x, mu, logvar, beta):
    BCE = F.binary_cross_entropy(recon, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

def train_and_evaluate(beta):
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"β={beta} | Epoch {epoch+1} | Avg Loss: {total_loss / len(train_loader.dataset):.4f}")

    model.eval()
    y_true, y_score = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            recon, _, _ = model(data)
            error = F.mse_loss(recon, data.view(-1, 784), reduction="none")
            error = error.mean(dim=1)
            y_score.extend(error.cpu().numpy())
            y_true.extend((target != 1).int().cpu().numpy())

    threshold = np.percentile(y_score, 99)
    y_pred = (np.array(y_score) > threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)

    return precision, recall, auc

betas = [1.0, 2.0]
results = {}

for beta in betas:
    print(f"\nTraining β-VAE with β = {beta}")
    results[beta] = train_and_evaluate(beta)

print("\n=== FINAL COMPARISON RESULTS ===")
for beta, (p, r, a) in results.items():
    print(f"β={beta} | Precision={p:.3f} | Recall={r:.3f} | AUC={a:.3f}")
