import json, numpy as np
import pandas as pd

def _fix_counts_to_sum(counts, total):
    diff = int(total) - int(np.sum(counts))
    if diff == 0:
        return counts.astype(int)
    idx = np.argsort(-np.modf(counts)[0])
    counts = counts.astype(int)
    i = 0
    while diff != 0 and i < len(counts):
        j = idx[i]
        if diff > 0:
            counts[j] += 1
            diff -= 1
        else:
            if counts[j] > 0:
                counts[j] -= 1
                diff += 1
        i += 1
    return counts

def sample_user_types_from_gmm(gmm_config: dict, K_types: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    comps = gmm_config["components"]
    d = int(gmm_config.get("dim", 2))
    w = np.array([c["w"] for c in comps], dtype=float)
    w = w / w.sum()
    mu = [np.array(c["mu"], dtype=float) for c in comps]
    cov = [np.array(c["cov"], dtype=float) for c in comps]
    raw_counts = w * K_types
    n_per = _fix_counts_to_sum(raw_counts, K_types)

    thetas = []
    pis = []

    for q, n_q in enumerate(n_per):
        n_q = int(max(1, n_q))
        samp = rng.multivariate_normal(mean=mu[q], cov=cov[q], size=n_q, method="cholesky")
        wt = w[q] / n_q
        thetas.append(samp)
        pis.append(np.full((n_q,), wt, dtype=float))
    thetas = np.vstack(thetas)
    pi = np.concatenate(pis)
    pi = pi / pi.sum()
    return thetas, pi, {"n_per_component": n_per.tolist(), "weights": w.tolist()}

def save_users_csv(path, thetas, pi):
    pd.DataFrame({"theta1": thetas[:,0], "theta2": thetas[:,1], "pi": pi}).to_csv(path, index=False)
    return path
