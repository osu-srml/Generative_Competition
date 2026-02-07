import numpy as np, pandas as pd
from routing import user_routing_n_platforms
from metrics import welfare_selected, diversity_from_shares
def best_response_actor(S, pi, f, actor_idx):
    M, K = S.shape
    P = len(f)
    best_idx, best_U = None, -1e18
    for m in range(M):
        f_try = list(f)
        f_try[actor_idx] = m
        S_sel = np.vstack([S[idx] for idx in f_try])
        _, U_try, _ = user_routing_n_platforms(S_sel, pi)
        if U_try[actor_idx] > best_U + 1e-18:
            best_U = U_try[actor_idx]
            best_idx = m
    return int(best_idx), float(best_U)

def run_best_response_sequence(S, pi, P, max_rounds=10, f_init=None):
    if f_init is None: 
        f = [0]*P
    else:
        assert len(f_init)==P
        f = list(f_init)
    hist = [] 
    step = 0
    S_sel = np.vstack([S[idx] for idx in f])
    _, U, shares = user_routing_n_platforms(S_sel, pi)

    W = welfare_selected(S_sel, pi)
    H, HHI = diversity_from_shares(shares)
    row = {"step": step, "round": 0, "actor": -1, "W": float(W), "shannon": float(H), "hhi": float(HHI)}
    row.update({f"f{i}": int(f[i]) for i in range(P)})
    row.update({f"U{i}": float(U[i]) for i in range(P)})
    hist.append(row)
    step += 1
    for rnd in range(1, max_rounds+1):
        changed = False
        for i in range(P):
            best_idx, _ = best_response_actor(S, pi, f, i)
            if best_idx != f[i]:
                f[i] = best_idx
                changed = True
            S_sel = np.vstack([S[idx] for idx in f])
            _, U, shares = user_routing_n_platforms(S_sel, pi)

            W = welfare_selected(S_sel, pi)
            H, HHI = diversity_from_shares(shares)

            row = {"step": step, "round": rnd, "actor": i, "W": float(W), "shannon": float(H), "hhi": float(HHI)}
            row.update({f"f{k}": int(f[k]) for k in range(P)})
            row.update({f"U{k}": float(U[k]) for k in range(P)})
            hist.append(row)
            step += 1
        if not changed: break
    return pd.DataFrame(hist)
