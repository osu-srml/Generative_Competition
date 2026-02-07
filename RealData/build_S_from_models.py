import argparse, os, json
import numpy as np, torch
from core.choice import read_models_list, read_users_mix, build_S
from core.reward import load_reward_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_list", required=True, type=str)
    ap.add_argument("--users_mix", required=True, type=str)
    ap.add_argument("--n_eval", type=int, default=2048)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--hub_entry", type=str, default="cifar10_resnet20")
    ap.add_argument("--reward_mode", type=str, default="probs", choices=["probs","acc"])
    ap.add_argument("--outdir", type=str, default="./outputs/S_build")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = read_models_list(args.models_list)
    user_weights, pi, user_names = read_users_mix(args.users_mix)
    reward_model = load_reward_model(args.hub_entry, pretrained=True, device=device)

    S, model_names = build_S(models, user_weights, reward_model, n_eval=args.n_eval, device=device,
                             timesteps=args.timesteps, 
#                             save_grids_dir=os.path.join(args.outdir,"grids"),
                             reward_mode=args.reward_mode
                             )

    np.save(os.path.join(args.outdir,"S.npy"), S)
    np.save(os.path.join(args.outdir,"pi.npy"), pi)
    json.dump(model_names, open(os.path.join(args.outdir,"model_names.json"),"w"))
    json.dump(user_names,  open(os.path.join(args.outdir,"user_names.json"),"w"))
    print(f"S shape={S.shape}, saved {args.outdir}")

if __name__ == "__main__":
    main()
