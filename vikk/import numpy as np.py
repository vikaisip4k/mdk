import numpy as np
a = np.array([3, 1, 4, 1, 5])
print(a * 2, a + 10, a ** 2)
print(a.mean(), a.min(), a.max(), a.sum())

M = np.arange(12).reshape(3, 4)
print(M, M.shape)
print('строка 0:', M[0]); print('столбец 1:', M[:, 1]); print('элемент:', M[2, 3])
print(M.T.shape)

print(M.mean(axis=0))
print(M.mean(axis=1))

A = np.array([[1, 2, 3], [4, 5, 6]])
w = np.array([[1], [0], [-1]])
print(A @ w)
print(A + np.array([10, 20, 30]))

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
w = np.array([0.5, -1.0, 2.0, 0.0, 1.5]); b = 0.3
y = X @ w + b
print(X.shape, w.shape, y.shape, y[:5])

with open('titanic.csv', encoding='utf-8') as f:
    header = f.readline().strip().split(',')
    rows = [line.strip().split(',') for line in f]

cols = {h: [] for h in header}
for r in rows:
    for h, v in zip(header, r):
        if v == '':
            continue
        try:
            cols [h].append(float(v))
        except ValueError:
            cols [h].append (v)

ages=np.array(cols['age'])
print (f'{h:12s} n={len(ages):4d} mean={ages.mean():8.2f}min={ages.min():6.1f} max={ages.max():6.1f};')