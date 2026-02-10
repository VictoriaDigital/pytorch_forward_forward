"""
Forward-Forward Algorithm - 1000 Epochs with Early Stopping
Experiment 3: Full paper settings + early stopping

Changes from main_full.py:
- 1000 epochs per layer (matching paper exactly)
- Early stopping: stop if loss doesn't improve for 50 epochs
- Loss logging for analysis
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import time
import psutil
import os
import json

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
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

    def train_layers(self, x_pos, x_neg, x_train, y_train):
        h_pos, h_neg = x_pos, x_neg
        all_logs = []
        for i, layer in enumerate(self.layers):
            print(f'\n[Layer {i+1}/{len(self.layers)}] Training {layer.in_features} → {layer.out_features}')
            start = time.time()
            h_pos, h_neg, layer_log = layer.train_layer(h_pos, h_neg, i, self, x_train, y_train)
            elapsed = time.time() - start
            all_logs.append(layer_log)
            print(f'  Completed in {elapsed:.1f}s ({elapsed/60:.1f} min) | Memory: {get_memory_mb():.0f}MB')
            print(f'  Final loss: {layer_log["final_loss"]:.4f} | Epochs run: {layer_log["epochs_run"]}')
            if layer_log["early_stopped"]:
                print(f'  ⚡ Early stopped at epoch {layer_log["epochs_run"]}')
        return all_logs


class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000  # Full paper setting
        self.patience = 50     # Early stopping patience

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train_layer(self, x_pos, x_neg, layer_idx, net, x_train, y_train):
        best_loss = float('inf')
        patience_counter = 0
        losses = []
        
        pbar = tqdm(range(self.num_epochs), desc="  Epochs", leave=False)
        for i in pbar:
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold
            ]))).mean()
            
            loss_val = loss.item()
            losses.append(loss_val)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            # Early stopping check
            if loss_val < best_loss - 1e-4:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Update progress bar
            if i % 10 == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "best": f"{best_loss:.4f}", "patience": patience_counter})
            
            if patience_counter >= self.patience:
                pbar.close()
                print(f'  Early stopping triggered at epoch {i+1}')
                break
        
        layer_log = {
            "layer": layer_idx,
            "epochs_run": i + 1,
            "early_stopped": patience_counter >= self.patience,
            "final_loss": loss_val,
            "best_loss": best_loss,
            "losses": losses[::10]  # Sample every 10th for space
        }
        
        return self.forward(x_pos).detach(), self.forward(x_neg).detach(), layer_log


if __name__ == "__main__":
    print("=" * 60)
    print("Forward-Forward Algorithm - EXPERIMENT 3")
    print("1000 Epochs + Early Stopping")
    print("=" * 60)
    print(f"Target: ~1.4% test error")
    print(f"Architecture: [784 → 2000 → 2000 → 2000 → 2000]")
    print(f"Training samples: 50,000 (full MNIST)")
    print(f"Epochs per layer: 1000 (with early stopping, patience=50)")
    print("=" * 60)
    
    torch.manual_seed(1234)
    start_time = time.time()
    
    print(f"\n[1] Loading MNIST... (Memory: {get_memory_mb():.0f}MB)")
    train_loader, test_loader = MNIST_loaders()
    
    print(f"\n[2] Creating network [784 → 2000 → 2000 → 2000 → 2000]...")
    net = Net([784, 2000, 2000, 2000, 2000])
    print(f"    Network created (Memory: {get_memory_mb():.0f}MB)")
    
    print(f"\n[3] Preparing data...")
    x, y = next(iter(train_loader))
    print(f"    Training samples: {x.shape[0]}")
    
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    print(f"    Positive/negative data ready (Memory: {get_memory_mb():.0f}MB)")
    
    print(f"\n[4] Training with Forward-Forward...")
    train_start = time.time()
    training_logs = net.train_layers(x_pos, x_neg, x, y)
    train_time = time.time() - train_start
    
    print(f"\n[5] Evaluating...")
    with torch.no_grad():
        train_acc = net.predict(x).eq(y).float().mean().item()
        train_error = 1.0 - train_acc
        print(f"    Train error: {train_error*100:.2f}%")
        
        x_te, y_te = next(iter(test_loader))
        test_acc = net.predict(x_te).eq(y_te).float().mean().item()
        test_error = 1.0 - test_acc
        print(f"    Test error:  {test_error*100:.2f}%")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("RESULTS - EXPERIMENT 3")
    print("=" * 60)
    print(f"Train error:    {train_error*100:.2f}%")
    print(f"Test error:     {test_error*100:.2f}%")
    print(f"Train time:     {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"Total time:     {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Peak memory:    {get_memory_mb():.0f}MB")
    print(f"Target error:   ~1.4% (paper)")
    print("=" * 60)
    
    # Per-layer summary
    print("\nPer-layer training summary:")
    total_epochs = 0
    for log in training_logs:
        status = "⚡ EARLY STOPPED" if log["early_stopped"] else "✓ completed"
        print(f"  Layer {log['layer']}: {log['epochs_run']} epochs, loss={log['final_loss']:.4f} {status}")
        total_epochs += log["epochs_run"]
    print(f"  Total epochs: {total_epochs} / {1000 * 4} possible")
    
    # Save results
    results = {
        "experiment": "3 - 1000 epochs + early stopping",
        "train_error": train_error * 100,
        "test_error": test_error * 100,
        "train_time_s": train_time,
        "total_time_s": total_time,
        "peak_memory_mb": get_memory_mb(),
        "total_epochs_run": total_epochs,
        "per_layer": training_logs
    }
    
    with open('results_exp3.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results_exp3.json")
