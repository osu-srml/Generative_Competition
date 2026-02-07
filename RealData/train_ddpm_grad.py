import os, math, json, argparse
import numpy as np
import torch, torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP

from train_ddpm import UNet32, DDPM, apply_lora_to_unet, set_lora_scale, unwrap, is_dist, is_main_process
from core.reward import load_reward_model, normalize_for_cifar, CIFAR10_LABELS
from core.choice import read_users_mix

def load_sbar_from_Sdir(S_dir: str, opponent_cols: str | None):
    S_full = np.load(os.path.join(S_dir, "S.npy"))      # [n_users, n_models]
    n_users, n_models = S_full.shape
    if opponent_cols:
        cols = sorted(set(int(x) for x in opponent_cols.split(",")))
        assert len(cols) > 0 and all(0 <= c < n_models for c in cols), f"cols out of range: {cols}"
        S_op = S_full[:, cols]                          # [n_users, |opponents|]
    else:
        S_op = S_full

    Sbar = S_op.max(axis=1).astype(np.float32)          # [n_users]
    return Sbar

def stabilize_scores(S_batch: torch.Tensor, mode: str, m0: float, eps: float = 1e-8):
    logs = {}
    if mode == "zscore":
        mean = S_batch.mean()
        std  = S_batch.std().clamp_min(eps)
        z = (S_batch - mean) / std
        logs.update({"S_mean": mean.item(), "S_std": std.item()})
        S_tilde = z - m0
    elif mode == "mad":
        med = S_batch.median()
        mad = (S_batch - med).abs().median().clamp_min(eps)
        z = (S_batch - (med)) / mad
        logs.update({"S_med": med.item(), "S_mad": mad.item()})
        S_tilde = z - m0
    else:
        S_tilde = S_batch - m0
    return S_tilde, logs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="./data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", type=str)
    ap.add_argument("--save", type=str, default="checkpoints/ddpm_grad.pt")
    ap.add_argument("--channels_last", action="store_true")

    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_scale", type=float, default=1.0)

    ap.add_argument("--users_mix", required=True)
    ap.add_argument("--S_dir", required=True, help="dir with S.npy")
    ap.add_argument("--opponent_cols", type=str, default=None, help="cols (models) used as opponents")
    ap.add_argument("--lambda_choice", type=float, default=0.05, help="weight of choice reward")

    ap.add_argument("--sample_every", type=int, default=50)
    args = ap.parse_args()

    # init
    need_dist = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
    if need_dist:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        try: torch.set_float32_matmul_precision("high")
        except: pass

    # data
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    download_flag = (not need_dist) or (torch.distributed.get_rank() == 0 if need_dist else True)
    train_dataset = datasets.CIFAR10(root=args.dataset_root, train=True, download=download_flag, transform=tf)
    if need_dist:
        torch.distributed.barrier()

    if need_dist:
        sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        sampler, shuffle = None, True

    loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    # model
    unet = UNet32(base_ch=64)
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        unet.load_state_dict(sd["model"], strict=False)

    if args.lora_rank and args.lora_rank > 0:
        unet = apply_lora_to_unet(unet, r=args.lora_rank, alpha=args.lora_alpha)
        for n,p in unet.named_parameters():
            if ("A.weight" in n) or ("B.weight" in n):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
    set_lora_scale(unet, args.lora_scale)

    if args.channels_last:
        unet.to(memory_format=torch.channels_last)

    ddpm = DDPM(unet, timesteps=args.timesteps).to(device)
    if need_dist:
        
        ddpm = DDP(ddpm, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    # reward
    clf = load_reward_model("cifar10_resnet20", pretrained=True, device=device).eval()
    for p in clf.parameters(): p.requires_grad_(False)

    user_weights, pi_np, user_names = read_users_mix(args.users_mix)
    pi_t = torch.tensor(pi_np, device=device, dtype=torch.float32)          # [K]
    Sbar_np = load_sbar_from_Sdir(args.S_dir, args.opponent_cols)           # [K]
    Sbar_t = torch.tensor(Sbar_np, device=device, dtype=torch.float32)      # [K]
    K = len(user_weights)
    assert pi_t.shape[0] == K == Sbar_t.shape[0], (pi_t.shape, K, Sbar_t.shape)

    opt = torch.optim.AdamW([p for p in ddpm.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    #beta = float(args.beta_choice)
    lam  = float(args.lambda_choice)

    # train
    for epoch in range(1, args.epochs+1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        unwrap(ddpm).train()
        for x,_ in loader:
            x = x.to(device, non_blocking=True)
            B = x.size(0)
            t = torch.randint(0, args.timesteps, (B,), device=device, dtype=torch.long)
            eps = torch.randn_like(x)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=True):

                x_t = unwrap(ddpm).q_sample(x, t, noise=eps)
                eps_pred = unwrap(ddpm).model(x_t, t.float())
                L_denoise = F.mse_loss(eps_pred, eps)


                alpha_bar_t = unwrap(ddpm).alphas_cumprod[t].view(-1,1,1,1)
                x0_hat = (x_t - torch.sqrt(1-alpha_bar_t)*eps_pred) / torch.sqrt(alpha_bar_t)


                x01 = (x0_hat + 1.0) / 2.0
                probs = clf(normalize_for_cifar(x01)).softmax(dim=1)  # [B,10], grad flows to x0_hat only

                # S_k
                S_list = []
                for gw in user_weights:
                    w = torch.tensor([gw.get(c, 0.0) for c in CIFAR10_LABELS], device=device, dtype=probs.dtype)
                    S_k = (probs * w.view(1,-1)).sum(1).mean()  # scalar
                    S_list.append(S_k)
                S_batch = torch.stack(S_list, dim=0)  # [K]

                beta = 4
                delta = S_batch - Sbar_t                         # [K]
                sigma = torch.sigmoid(beta * delta)
                #L_choice = (pi_t * delta).sum()
                #loss = L_denoise - lam * L_choice
                L_choice = (pi_t * sigma * S_batch).sum()
                loss = L_denoise - lam * L_choice 

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # sample
        if (epoch % args.sample_every == 0 or epoch == args.epochs) and is_main_process():
            core = unwrap(ddpm)
            with torch.no_grad():
                samples = core.sample((64,3,32,32), device)
                grid = vutils.make_grid((samples+1)/2, nrow=8)
                os.makedirs(os.path.dirname(args.save), exist_ok=True)
                vutils.save_image(grid, os.path.join(os.path.dirname(args.save), f"samples_choice_e{epoch:03d}.png"))
            torch.save({"model": core.model.state_dict()}, args.save)
            print(f"saved {args.save}")

        if is_main_process():
            with torch.no_grad():
                delta = (S_batch - Sbar_t).detach().cpu()
                d_mean = delta.mean().item()
                d_min = delta.min().item()
                d_max = delta.max().item()
                print(f"Epoch {epoch:03d} Ld={L_denoise.item():.4f} "
                      f"Lc={L_choice.item():.4f} loss={loss.item():.4f} "
                      f"d_mean={d_mean:.4f} d_min={d_min:.4f} d_max={d_max:.4f}")
            

    if need_dist:
        dist.destroy_process_group()
    if is_main_process():
        print("Over")

if __name__ == "__main__":
    main()