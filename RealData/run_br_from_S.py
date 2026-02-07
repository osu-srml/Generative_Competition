import argparse, os, json
import numpy as np
import pandas as pd
from core.choice import run_best_response
from core.plots import plot_steps, heatmap_U, heatmap_WD, coverage_heatmap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S_dir", required=True, type=str)
    ap.add_argument("--players", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--outdir", type=str, default="./outputs/BR")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    S = np.load(os.path.join(args.S_dir,"S.npy"))
    pi = np.load(os.path.join(args.S_dir,"pi.npy"))
    model_names = json.load(open(os.path.join(args.S_dir,"model_names.json")))
    user_names  = json.load(open(os.path.join(args.S_dir,"user_names.json")))

    M = S.shape[1]
    pool = list(range(M))
    df = run_best_response(S, pi, P=args.players, rounds=args.rounds, model_pool=pool)
    df.to_csv(os.path.join(args.outdir,"history.csv"), index=False)

    plot_steps(df, os.path.join(args.outdir,"steps.png"))
    heatmap_U(df, os.path.join(args.outdir,"heatmap_U.png"), P=args.players)

    # final chosen
    per = df.groupby("round").tail(1)
    chosen = list(map(int, per.iloc[-1]["chosen"].split(",")))

    heatmap_WD(df, os.path.join(args.outdir,"heatmap_W_D.png"))
    coverage_heatmap(S, pi, chosen, model_names, user_names, os.path.join(args.outdir,"heatmap_coverage.png"))
    print(f"BR results {args.outdir}")

if __name__ == "__main__":
    main()
