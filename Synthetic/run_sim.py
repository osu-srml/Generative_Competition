import os, json, argparse
import numpy as np
import pandas as pd
from users import sample_user_types_from_gmm, save_users_csv
from models import load_models_from_json, save_models_csv
from metrics import scores_matrix, compute_T, compute_delta
from br import run_best_response_sequence

def ensure_dir(d): 
    os.makedirs(d, exist_ok=True)
    return d

def main():
    ap = argparse.ArgumentParser(description="Minimal BR simulator")
    ap.add_argument("--users", type=str, default="configs/users_gmm.json", help="path to GMM config json")
    ap.add_argument("--models", type=str, default="configs/models.json", help="path to models json")
    ap.add_argument("--players", type=str, default="configs/players.json", help="path to players json")
    ap.add_argument("--K_types", type=int, default=12, help="number of discrete user types to generate from GMM")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rounds", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="./output")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    gmm_cfg = json.load(open(args.users, "r", encoding="utf-8"))
    thetas, pi, meta_u = sample_user_types_from_gmm(gmm_cfg, K_types=args.K_types, seed=args.seed)
    save_users_csv(os.path.join(outdir, "users.csv"), thetas, pi)

    models = load_models_from_json(args.models)
    save_models_csv(os.path.join(outdir, "models.csv"), models)

    players = json.load(open(args.players, "r", encoding="utf-8"))
    P = int(players.get("num_players", 2))
    names = players.get("names", [f"Player {i}" for i in range(P)])

    S = scores_matrix(models, thetas)      # [M,K]
    T = compute_T(S, pi)                   # [M]
    delta = compute_delta(S, pi)           # [M,M]

    hist = run_best_response_sequence(S, pi, P=P, max_rounds=args.max_rounds, f_init=[0]*P)

    model_names = [m.name for m in models]
    pd.DataFrame(S, index=model_names, columns=[f"theta{k}" for k in range(thetas.shape[0])]).to_csv(os.path.join(outdir, "S.csv"))
    pd.DataFrame({"model": model_names, "T": T}).to_csv(os.path.join(outdir, "T.csv"), index=False)
    pd.DataFrame(delta, index=model_names, columns=model_names).to_csv(os.path.join(outdir, "delta.csv"))
    hist.to_csv(os.path.join(outdir, "history.csv"), index=False)

    summ = {"P": P, "players": names, "K_types": int(thetas.shape[0]), "users_meta": meta_u,
            "models": model_names, "rounds_logged": int(hist["round"].max()) if len(hist)>0 else 0}
    json.dump(summ, open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8"), indent=2)

    print(json.dumps({"outdir": outdir, "P": P, "K_types": int(thetas.shape[0]),
                      "models": model_names, "history_rows": int(len(hist))}, ensure_ascii=False))

if __name__ == "__main__":
    main()
