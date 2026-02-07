import numpy as np
def user_routing_n_platforms(S_selected, pi, tie_tol=1e-12):
    P, K = S_selected.shape
    max_per_k = S_selected.max(axis=0)
    ties = np.abs(S_selected - max_per_k[None, :]) <= tie_tol
    tie_counts = ties.sum(axis=0)
    P_share = ties / np.maximum(tie_counts, 1)[None, :]
    U = (pi[None, :] * P_share * S_selected).sum(axis=1)
    shares = (pi[None, :] * P_share).sum(axis=1)
    return P_share, U, shares
