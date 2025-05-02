import numpy as np

def genData(num_data, num_features, num_items, seed=42, Q=100, dim=1, deg=1, noise_width=0.5, epsilon=0.1):
    rnd = np.random.RandomState(seed)
    n = num_data
    p = num_features
    m = num_items
    
    x = rnd.normal(0, 1, (n, m, p))
    B = rnd.binomial(1, 0.5, (m, p))

    c = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            values = (np.dot(B[j], x[i, j].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
            values *= 5
            values /= 3.5 ** deg
            epislon = rnd.uniform(1 - noise_width, 1 + noise_width, 1)
            values *= epislon
            values = np.ceil(values)
            c[i, j] = values

    c = c.astype(np.float64)
    
    w = rnd.normal(0, 1, (m, p))
    b = rnd.normal(0, 1, (n, m))
    r = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            r[i, j] = np.dot(w[j], x[i, j]) + b[i, j]

    r = 1 / (1 + np.exp(-r))

    return x, r, c, Q
