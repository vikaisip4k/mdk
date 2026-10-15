import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons  # Заменили на make_moons

# Генерация изогнутого датасета (Шаг 6)
X, y = make_moons(200, noise=0.2, random_state=1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2)
b = 0.0

def forward(X): 
    return sig(X @ w + b)

print(bce := lambda p, y: -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
print(bce(forward(X), y))

lr, hist = 0.1, []
for i in range(500):
    p = forward(X)
    dz = p - y
    w -= lr * X.T @ dz / len(X)
    b -= lr * dz.mean()
    hist.append(bce(p, y))

plt.plot(hist)
plt.show()

print('accuracy:', ((forward(X) > 0.5) == y).mean())

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200), np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
