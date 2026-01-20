import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MLP(nn.Module):
    def __init__(self, in_dim=17, hidden=32):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)
        self.a = nn.ReLU()
    def forward(self, x):
        h1 = self.a(self.l1(x))
        h2 = self.a(self.l2(h1))
        out = self.l3(h2)
        return out, h1, h2

def train_surrogate(X, y, seed=42, epochs=300, lr=1e-3, snapshot_stride=10):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    Xs_tr = scaler.fit_transform(X_train).astype(np.float32)
    Xs_te = scaler.transform(X_test).astype(np.float32)
    ytr = y_train.astype(np.float32)
    yte = y_test.astype(np.float32)
    model = MLP(in_dim=X.shape[1], hidden=32)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(Xs_tr)
    yt = torch.tensor(ytr)
    Xv = torch.tensor(Xs_te)
    yv = torch.tensor(yte)
    snapshots = []
    for ep in range(epochs):
        opt.zero_grad()
        pred, _, _ = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        if ep % snapshot_stride == 0 or ep == epochs - 1:
            w1 = model.l1.weight.detach().cpu().numpy()
            w2 = model.l2.weight.detach().cpu().numpy()
            w3 = model.l3.weight.detach().cpu().numpy()
            snapshots.append((w1, w2, w3))
    return model, scaler, snapshots

def predict_surrogate(model, scaler, X):
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        out, _, _ = model(torch.tensor(Xs))
    return out.squeeze().detach().cpu().numpy()

def gradient_sensitivity(model, scaler, x):
    xs = scaler.transform(x.reshape(1, -1)).astype(np.float32)
    t = torch.tensor(xs, requires_grad=True)
    out, _, _ = model(t)
    out.backward(torch.ones_like(out))
    g = t.grad.squeeze().detach().cpu().numpy()
    return np.abs(g)
