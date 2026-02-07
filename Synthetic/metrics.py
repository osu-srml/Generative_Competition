import numpy as np, pandas as pd
def scores_matrix(models, thetas):
    return np.vstack([m.score(thetas) for m in models])

def compute_T(S, pi):
    return S @ pi

def Z_pair(Si, Sj):
    Z = np.where(Si > Sj, Si, 0.0)
    Z = np.where(Si < Sj, -Si, Z)
    return Z

def compute_delta(S, pi):
    M, K = S.shape
    delta = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(M):
            if i == j: continue
            delta[i, j] = float(np.dot(pi, Z_pair(S[i], S[j])))
    return delta

def welfare_selected(S_selected, pi):
    return float((pi * S_selected.max(axis=0)).sum())

def diversity_from_shares(shares):
    p = np.array(shares, dtype=float)
    s = p.sum()
    if s > 0:
        p = p/s
    H = float(-(p*np.log(p+1e-12)).sum())
    HHI = float((p**2).sum())
    return H, HHI
