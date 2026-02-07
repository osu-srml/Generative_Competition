import os, math, json, torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from torchvision.utils import make_grid, save_image

from .generator import load_generator
from .reward import reward_scores

def read_models_list(path: str) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def read_users_mix(path: str) -> Tuple[List[Dict[str,float]], np.ndarray, List[str]]:
    cfg = json.load(open(path, "r"))
    classes = cfg["classes"]
    groups = cfg["groups"]
    weights = [g["weights"] for g in groups]
    pi = np.array([g.get("mass", 1.0/len(groups)) for g in groups], dtype=np.float64)
    pi = pi / pi.sum()
    names = [g.get("name", f"G{i}") for i,g in enumerate(groups)]
    return weights, pi, names

@torch.no_grad()
def build_S(models, user_weights, reward_model, n_eval,
            device, timesteps=1000, save_grids_dir=None, reward_mode="probs",
            sample_chunk=256, cls_chunk=256, use_amp=True):
    
    K = len(user_weights)
    M = len(models)
    S = np.zeros((K, M), dtype=np.float64)
    model_names = []

    for j, ck in enumerate(models):
        name = os.path.splitext(os.path.basename(ck))[0]
        model_names.append(name)

        gen = load_generator(ck, device=device, timesteps=timesteps, try_lora=True)

        imgs_list = []
        remain = n_eval
        while remain > 0:
            bs = min(sample_chunk, remain)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    xs = gen.sample(bs, device=device)
            else:
                xs = gen.sample(bs, device=device)
            imgs_list.append(xs.cpu())
            remain -= bs
            print(f"[eval] sampling {bs} / remaining {remain}", flush=True)
        imgs = torch.cat(imgs_list, dim=0)  # [n_eval,3,32,32]

        if save_grids_dir:
            os.makedirs(save_grids_dir, exist_ok=True)
            nrow = int(max(1, round(n_eval ** 0.5)))
            grid = torchvision.utils.make_grid((imgs[:nrow*nrow] + 1) / 2, nrow=nrow)
            torchvision.utils.save_image(grid, os.path.join(save_grids_dir, f"{j:02d}_{name}.png"))

        imgs = imgs.to(device, non_blocking=True)
        for k, gw in enumerate(user_weights):
            scores = []
            for s in range(0, n_eval, cls_chunk):
                chunk = imgs[s:s+cls_chunk]
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        r = reward_scores(chunk, reward_model, gw, mode=reward_mode)  # [B]
                else:
                    r = reward_scores(chunk, reward_model, gw, mode=reward_mode)
                scores.append(r.float())
            S[k, j] = torch.cat(scores).mean().item()

        del gen, imgs, imgs_list, xs, r, scores
        torch.cuda.empty_cache()

    return S, model_names

def hard_choice_assignment(S: np.ndarray, pi: np.ndarray, chosen_models: List[int]) -> Tuple[np.ndarray, float]:
    sub = S[:, chosen_models]        # K x P
    max_scores = sub.max(axis=1, keepdims=True)       # K x 1
    winners = (sub == max_scores)                     # K x P (bool mask)
    shares = winners / winners.sum(axis=1, keepdims=True)  # K x P, each winner gets 1/#winners
    # pi_k * S_kj * share_kj
    payoffs = (pi[:,None] * sub * shares).sum(axis=0)        # shape: (P,)
    welfare = float((pi * max_scores[:,0]).sum())
    return payoffs, welfare

def market_shares(win: np.ndarray, pi: np.ndarray, P: int) -> np.ndarray:
    shares = np.zeros(P, dtype=np.float64)
    for k, w in enumerate(win):
        shares[w] += pi[k]
    return shares

def assign_with_tie_split(S: np.ndarray, pi: np.ndarray, chosen_models: List[int]):
    sub = S[:, chosen_models]             # K x P
    max_scores = sub.max(axis=1, keepdims=True)  # K x 1
    winners = (sub == max_scores)         # K x P (bool)
    tie_counts = winners.sum(axis=1, keepdims=True).astype(np.float64)  # K x 1
    shares = winners / tie_counts         # K x P

    mass = (pi[:, None] * shares).sum(axis=0)              # (P,)
    U = (pi[:, None] * shares * sub).sum(axis=0)           # (P,)
    W = float((pi * max_scores[:, 0]).sum())               # scalar
    return U, mass, W

def entropy_shannon(mass: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.sum([m * math.log(m + eps) for m in mass]))

def welfare_full(S: np.ndarray, pi: np.ndarray) -> float:
    return float((pi * S.max(axis=1)).sum())

def run_best_response(S: np.ndarray, pi: np.ndarray, P: int, rounds: int, model_pool: List[int]) -> pd.DataFrame:
    chosen = [model_pool[0] for _ in range(P)]
    rows = []
    step = 0
    W_full = welfare_full(S, pi)

    for r in range(rounds + 1):
        # actor = -1
        U_vec, mass_vec, W = assign_with_tie_split(S, pi, chosen)
        hhi = float((mass_vec ** 2).sum())
        shannon = entropy_shannon(mass_vec)

        for i in range(P):
            rows.append(dict(
                step=step, round=r, actor=-1, player=i,
                U=float(U_vec[i]),
                mass=float(mass_vec[i]),
                W=W, shannon=shannon, hhi=hhi,
                chosen=",".join(map(str, chosen))
            ))
        step += 1

        if r == rounds:
            break

        for i in range(P):
            best_val = -1e18
            best_m = chosen[i]
            best_U_vec = None
            best_mass_vec = None
            best_W = None

            for m in model_pool:
                cand = chosen.copy()
                cand[i] = m
                U2, mass2, W2 = assign_with_tie_split(S, pi, cand)

                if U2[i] > best_val:
                    best_val = float(U2[i])
                    best_m = m
                    best_U_vec = U2
                    best_mass_vec = mass2
                    best_W = W2

            chosen[i] = best_m
            hhi2 = float((best_mass_vec ** 2).sum())
            shannon2 = entropy_shannon(best_mass_vec)

            rows.append(dict(
                step=step, round=r+1, actor=i, player=i,
                U=float(best_U_vec[i]),
                mass=float(best_mass_vec[i]),
                W=best_W, W_full=W_full,
                shannon=shannon2, hhi=hhi2,
                chosen=",".join(map(str, chosen))
            ))
            step += 1

    return pd.DataFrame(rows)