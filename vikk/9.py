from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from sklearn.datasets import make_moons
import pandas as pd
rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 1, 30)); y = np.sin(2 * np.pi * X) + rng.normal(0, .3, 30)
Xv = np.sort(rng.uniform(0, 1, 30)); yv = np.sin(2 * np.pi * Xv) + rng.normal(0, .3, 30)
for d in [1, 4, 15]:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(X[:, None], y)
    print(d, np.mean((m.predict(X[:, None]) - y)**2), np.mean((m.predict(Xv[:, None]) - yv)**2))
print('----------------------')

def fit(model, opt, steps=2000):
    tr, va = [], []
    for i in range(steps):
        model.train(); opt.zero_grad(); l = loss_fn(model(Xtr).squeeze(1), ytr); l.backward(); opt.step()
        if i % 50 == 0:
            model.eval()
            with torch.no_grad(): tr.append(l.item()); va.append(loss_fn(model(Xv).squeeze(1), yv).item())
    return tr, va
Xm, ym = make_moons(100, noise=0.3, random_state=1)
X_t = torch.tensor(Xm, dtype=torch.float32)   # без reshape!
y_t = torch.tensor(ym, dtype=torch.float32)
Xtr, ytr = X_t[:50], y_t[:50]
Xv, yv   = X_t[50:], y_t[50:]
loss_fn = nn.MSELoss()
big = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
def fit_early_stop(model, opt, steps=2000, patience=10):
    tr, va = [], []
    best_loss, best_w, wait = float('inf'), None, 0

    for i in range(steps):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr).squeeze(1), ytr)
        loss.backward()
        opt.step()

        if i % 50 == 0:
            model.eval()
            with torch.no_grad():
                v_loss = loss_fn(model(Xv).squeeze(1), yv).item()
            tr.append(loss.item())
            va.append(v_loss)

            if v_loss < best_loss:
                best_loss, best_w, wait = v_loss, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Стоп на шаге {i}. Лучший val loss: {best_loss:.4f}")
                    break

    if best_w:
        model.load_state_dict(best_w)
    return tr, va
tr, va = fit_early_stop(big, torch.optim.Adam(big.parameters(), 1e-2))
plt.plot(tr, label='train'); plt.plot(va, label='val'); plt.legend(); plt.title('Шаг 3: Early Stop'); plt.show()
print('-----------------------')

big_wd = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr4, va4 = fit_early_stop(
    big_wd,
    torch.optim.Adam(big_wd.parameters(), 1e-2, weight_decay=1e-2),
    patience=9999  
)
plt.plot(tr4, label='train'); plt.plot(va4, label='val')
plt.legend(); plt.title('Шаг 4: Weight Decay'); plt.show()
print('-------------------------')

reg = nn.Sequential(
    nn.Linear(2, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 1)
)
tr5, va5 = fit_early_stop(
    reg, 
    torch.optim.Adam(reg.parameters(), 1e-2, weight_decay=1e-2),
    patience=10
)
plt.plot(tr5, label='train'); plt.plot(va5, label='val')
plt.legend(); plt.title('Шаг 5: Dropout + Weight Decay'); plt.show()
print('-------------------------')

Xm_big, ym_big = make_moons(2000, noise=0.3, random_state=1)
X_big = torch.tensor(Xm_big, dtype=torch.float32)
y_big = torch.tensor(ym_big, dtype=torch.float32)
Xtr_big, ytr_big = X_big[:1000], y_big[:1000]
Xv_big, yv_big = X_big[1000:], y_big[1000:]

Xtr, ytr, Xv, yv = Xtr_big, ytr_big, Xv_big, yv_big
big_data = nn.Sequential(
    nn.Linear(2, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 1)
)

tr6, va6 = fit_early_stop(
    big_data, 
    torch.optim.Adam(big_data.parameters(), 1e-2),
    patience=10
)

plt.plot(tr6, label='train'); plt.plot(va6, label='val')
plt.legend(); plt.title('Шаг 6: Больше данных (2000 точек)'); plt.show()
print('-------------------------')

def
plt.show
plt.show


get_best_val(va_list):
    return round(min(va_list), 4)

def get_final_gap(tr_list, va_list):
    return round(np.mean(va_list[-5:]) - np.mean(tr_list[-5:]), 4)
results = {
    'Метод': [
        '1. Early Stopping (Шаг 3)',
        '2. Weight Decay (Шаг 4)',
        '3. Dropout + WD (Шаг 5)',
        '4. Больше данных (Шаг 6)'
    ],
    'Лучший Val Loss': [
        get_best_val(va),
        get_best_val(va4),
        get_best_val(va5),
        get_best_val(va6)
    ],
    'Разрыв в конце': [
        get_final_gap(tr, va),
        get_final_gap(tr4, va4),
        get_final_gap(tr5, va5),
        get_final_gap(tr6, va6)
    ]
}

df = pd.DataFrame(results)
print("\n" + "="*60)
print(df.to_string(index=False))
print("="*60)
df.to_csv('overfit_results.csv', index=False)
print("Таблица сохранена в overfit_results.csv")