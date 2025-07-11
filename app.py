import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

# Model must match training architecture
class CVAE(torch.nn.Module):
    def __init__(self, label_dim=10, latent_dim=20):
        super(CVAE, self).__init__()
        self.fc1 = torch.nn.Linear(784 + label_dim, 400)
        self.fc21 = torch.nn.Linear(400, latent_dim)
        self.fc22 = torch.nn.Linear(400, latent_dim)
        self.fc3 = torch.nn.Linear(latent_dim + label_dim, 400)
        self.fc4 = torch.nn.Linear(400, 784)

    def encode(self, x, c):
        x = torch.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z = torch.cat([z, c], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c)

# Load model
device = torch.device("cpu")
model = CVAE().to(device)
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

st.title("MNIST Digit Generator")
digit = st.selectbox("Choose a digit (0–9)", list(range(10)))

if st.button("Generate Images"):
    c = torch.eye(10)[digit].unsqueeze(0).repeat(5, 1)
    z = torch.randn(5, 20)
    with torch.no_grad():
        samples = model.decode(z, c).view(-1, 1, 28, 28)

    grid = make_grid(samples, nrow=5, padding=2)
    # Convert grid tensor to numpy and format properly
    grid = grid.squeeze(0).numpy()  # (H, W)
    grid = (grid * 255).astype(np.uint8)
    grid = np.expand_dims(grid, axis=-1)  # Now shape is (H, W, 1)

    grid_rgb = np.repeat(grid, 3, axis=-1)  # (H, W, 3)
    imgs = samples.detach().cpu().numpy()  # (5, 1, 28, 28)
    imgs = imgs.squeeze(1)  # (5, 28, 28)
    imgs = (imgs * 255).astype(np.uint8)
    imgs_rgb = np.stack([np.stack([img]*3, axis=-1) for img in imgs])  # (5, 28, 28, 3)

    captions = [f"{digit}" for _ in range(5)]
    st.image(imgs_rgb, caption=captions, width=150)


