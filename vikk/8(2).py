import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X_raw, y_raw = make_moons(n_samples=500, noise=0.2, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

Xt = torch.tensor(X_scaled, dtype=torch.float32)
yt = torch.tensor(y_raw, dtype=torch.float32)

def train_model(hidden_neurons, steps=500, lr=0.1):
    model = nn.Sequential(
        nn.Linear(2, hidden_neurons),
        nn.ReLU(),
        nn.Linear(hidden_neurons, 1)
    )
    
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    for _ in range(steps):
        opt.zero_grad()
        pred = model(Xt).squeeze(1)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        
    with torch.no_grad():
        final_pred = torch.sigmoid(model(Xt).squeeze(1))
        classes = (final_pred >= 0.5).float()
        accuracy = (classes == yt).float().mean().item()
        
    return model, accuracy


_, acc_2 = train_model(2)
print(f"Neurons:  2 | Accuracy: {acc_2:.4f}")

_, acc_4 = train_model(4)
print(f"Neurons:  4 | Accuracy: {acc_4:.4f}")

_, acc_64 = train_model(64)
print(f"Neurons: 64 | Accuracy: {acc_64:.4f}")
