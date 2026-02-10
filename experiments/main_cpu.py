"""
Forward-Forward Algorithm - CPU Version
Based on mpezeshki/pytorch_forward_forward
Modified for CPU-only execution
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import time

# Use smaller batch for CPU
def MNIST_loaders(train_batch_size=5000, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """Replace first 10 pixels with one-hot label"""
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(Layer(dims[d], dims[d + 1]))

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(h.pow(2).mean(1))
            goodness_per_label.append(sum(goodness).unsqueeze(1))
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train_layers(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print(f'Training layer {i}...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 500  # Reduced for faster testing

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train_layer(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs), desc="  Epochs"):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold
            ]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


if __name__ == "__main__":
    print("=" * 50)
    print("Forward-Forward Algorithm - CPU Test")
    print("=" * 50)
    
    torch.manual_seed(1234)
    start_time = time.time()
    
    print("\n[1] Loading MNIST...")
    train_loader, test_loader = MNIST_loaders()
    
    print("\n[2] Creating network [784 -> 500 -> 500]...")
    net = Net([784, 500, 500])
    
    print("\n[3] Preparing data...")
    x, y = next(iter(train_loader))
    print(f"    Training samples: {x.shape[0]}")
    
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    
    print("\n[4] Training with Forward-Forward...")
    train_start = time.time()
    net.train_layers(x_pos, x_neg)
    train_time = time.time() - train_start
    
    print(f"\n[5] Evaluating...")
    with torch.no_grad():
        train_error = 1.0 - net.predict(x).eq(y).float().mean().item()
        print(f"    Train error: {train_error*100:.2f}%")
        
        x_te, y_te = next(iter(test_loader))
        test_error = 1.0 - net.predict(x_te).eq(y_te).float().mean().item()
        print(f"    Test error:  {test_error*100:.2f}%")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Train error: {train_error*100:.2f}%")
    print(f"Test error:  {test_error*100:.2f}%")
    print(f"Train time:  {train_time:.1f}s")
    print(f"Total time:  {total_time:.1f}s")
    print("=" * 50)
