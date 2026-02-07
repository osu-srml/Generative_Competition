import os, json, argparse
import numpy as np
import pandas as pd
import torch

from core.generator import load_generator
from core.reward import load_reward_model, normalize_for_cifar, CIFAR10_LABELS
from core.choice import read_users_mix   # read users_mix.json

@torch.no_grad()
def eval_S_ours_per_type(ckpt, users_mix_path, n_eval, timesteps, hub_entry, device, amp=True, chunk=256):
    gen = load_generator(ckpt, device=device, timesteps=timesteps, try_lora=True)
    xs = []
    left = n_eval
    while left > 0:
        b = min(chunk, left)
        with torch.amp.autocast(device_type='cuda', enabled=amp):
            x = gen.sample(b, device=device)  # [-1,1]
        xs.append(x.cpu())
        left -= b
        print(f"[eval] sampling {b} / remaining {left-b}", flush=True)
    imgs = torch.cat(xs, 0).to(device)

    clf = load_reward_model(hub_entry, pretrained=True, device=device).eval()
    x01 = (imgs + 1.0) / 2.0
    xnor = normalize_for_cifar(x01)
    with torch.amp.autocast(device_type='cuda', enabled=amp):
        probs = clf(xnor).softmax(dim=1)  # [N,10]

    user_weights, _, _ = read_users_mix(users_mix_path)  # list of dicts class->weight
    S_ours = np.zeros(len(user_weights), dtype=np.float64)
    for k, gw in enumerate(user_weights):
        w = torch.tensor([gw.get(c, 0.0) for c in CIFAR10_LABELS], device=probs.device, dtype=probs.dtype)
        S_ours[k] = float((probs * w.view(1,-1)).sum(1).mean().item())
    return S_ours

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S_dir", required=True)
    ap.add_argument("--opponent_cols", type=str, default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--users_mix", required=True)
    ap.add_argument("--n_eval", type=int, default=1024)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--hub_entry", type=str, default="cifar10_resnet20")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--beta_margin", type=float, default=4.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--adaptive_scale", action="store_true")
    ap.add_argument("--soft", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    S_full = np.load(os.path.join(args.S_dir, "S.npy"))      # [K,M]
    pi     = np.load(os.path.join(args.S_dir, "pi.npy"))     # [K]
    classes_path = os.path.join(args.S_dir, "classes.json")
    if os.path.exists(classes_path):
        classes = json.load(open(classes_path))
    else:
        classes = CIFAR10_LABELS

    if args.opponent_cols:
        cols = [int(x) for x in args.opponent_cols.split(",")]
        S_op = S_full[:, cols] 
    else:
        S_op = S_full
    Sbar = S_op.max(axis=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_ours = eval_S_ours_per_type(args.ckpt, args.users_mix, args.n_eval,
                                  args.timesteps, args.hub_entry, device=device)
    print("Eval: S_ours:", S_ours.shape)

    delta = S_ours - Sbar
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    if args.adaptive_scale:
        c = np.median(delta)
        mad = np.median(np.abs(delta - c)) + 1e-8
        #z   = (delta - (c + args.m0)) / mad
        z   = (delta - (c)) / mad
        p_win = sig(z)

        Sc = np.median(Sbar)
        Smad= np.median(np.abs(Sbar - Sc)) + 1e-8
        Sad = (Sbar-(Sc)) / Smad

        alpha = pi * (p_win ** args.gamma) * Sbar

    elif args.soft:
        S_all = np.concatenate([S_op, S_ours[:, None]], axis=1)  # [K, M_op+1]
        # softmax
        exp_scores = np.exp(args.beta * (S_all - S_all.max(axis=1, keepdims=True)))
        probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-12)
        p_win = probs[:, -1]   # [K]
        alpha = pi * (p_win ** args.gamma)
        alpha = alpha / (alpha.sum() + 1e-12)   
    else:
        p_win = sig(args.beta * delta)
        g_adv = sig(args.beta_margin * delta)
        alpha = pi * (p_win ** args.gamma) * (g_adv ** args.eta)

    alpha = alpha / (alpha.sum() + 1e-12)
   
    users_cfg = json.load(open(args.users_mix))
    groups = users_cfg["groups"]
    class_weights = {c: 0.0 for c in classes}
    for k, g in enumerate(groups):
        wk = g["weights"]
        for c in classes:
            class_weights[c] += float(alpha[k]) * float(wk.get(c, 0.0))

    tot = sum(class_weights.values()) + 1e-12
    for c in classes:
        class_weights[c] = max(1e-9, class_weights[c] / tot)

    z = sum(class_weights.values())
    for c in classes: class_weights[c] /= z

    json.dump(dict(
        beta=args.beta, beta_margin=args.beta_margin, gamma=args.gamma, eta=args.eta,
        p_win=p_win.tolist(), delta=delta.tolist(), alpha=alpha.tolist(),
        class_weights=class_weights, classes=classes, pi=pi.tolist()
    ), open(os.path.join(args.outdir, "choice_weights.json"), "w"), indent=2)

    pd.DataFrame({
        "S_ours": S_ours, "Sbar_opponent": Sbar, "delta": delta,
        "p_win": p_win, "alpha": alpha, "pi": pi
    }).to_csv(os.path.join(args.outdir, "per_type.csv"), index=False)

    print("Save:", os.path.join(args.outdir, "choice_weights.json"))

if __name__ == "__main__":
    main()