import argparse, os, math, json
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, DistributedSampler, WeightedRandomSampler
from torchvision import datasets, transforms, utils as vutils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# DDP helpers
def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not is_dist()) or dist.get_rank() == 0

def unwrap(m):
    return m.module if isinstance(m, DDP) else m

# Weighted Samplers
def _build_weight_tensors(dataset, class_weights_json):
    info = json.load(open(class_weights_json))
    classes = info.get("classes", None)
    cw = info["class_weights"]
    if classes is None:
        classes = CIFAR10_LABELS
    label2w = {i: float(cw[classes[i]]) for i in range(len(classes))}
    ws = torch.zeros(len(dataset), dtype=torch.float)
    for i in range(len(dataset)):
        _, y = dataset[i]
        ws[i] = label2w[int(y)]
    s = ws.sum()
    if s > 0:
        ws /= s
    return ws

class WeightedDistributedSampler(torch.utils.data.Sampler):
    def __init__(self, weights, num_samples_per_rank, replacement=True, seed=0, rank=None, world_size=None):
        self.weights = weights.float()
        self.replacement = replacement
        self.seed = seed
        if rank is None:
            rank = dist.get_rank() if is_dist() else 0
        if world_size is None:
            world_size = dist.get_world_size() if is_dist() else 1
        self.rank, self.world_size = rank, world_size
        self.num_samples_per_rank = int(num_samples_per_rank)
        self.epoch = 0
    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __iter__(self):
        g = torch.Generator(device='cpu')
        g.manual_seed(self.seed + self.epoch)
        total = self.num_samples_per_rank * self.world_size
        idx = torch.multinomial(self.weights, total, self.replacement, generator=g)
        idx = idx.view(self.world_size, -1)[self.rank]
        return iter(idx.tolist())
    def __len__(self): return self.num_samples_per_rank

class QuotaDistributedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, class_weights_json, num_samples_per_rank,
                 seed=0, rank=None, world_size=None):

        info = json.load(open(class_weights_json))
        classes = info.get("classes", CIFAR10_LABELS)
        cw = info["class_weights"]
        self.class_weight_dict = {i: float(cw[classes[i]]) for i in range(len(classes))}
        self.dataset = dataset
        self.num_samples_per_rank = num_samples_per_rank
        self.seed = seed

        if rank is None:
            rank = dist.get_rank() if is_dist() else 0
        if world_size is None:
            world_size = dist.get_world_size() if is_dist() else 1
        self.rank, self.world_size = rank, world_size

        self.class_to_indices = {}
        for idx in range(len(dataset)):
            _, y = dataset[idx]
            self.class_to_indices.setdefault(int(y), []).append(idx)

        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        total_samples = self.num_samples_per_rank * self.world_size

        weight_sum = sum(self.class_weight_dict.values())
        norm_weights = {k: v / weight_sum for k, v in self.class_weight_dict.items()}

        all_indices = []
        for cls, w in norm_weights.items():
            quota = int(round(total_samples * w))
            candidates = self.class_to_indices[cls]
            if len(candidates) == 0:
                continue
            sampled = torch.tensor(candidates)[torch.randint(
                low=0, high=len(candidates), size=(quota,), generator=g
            )].tolist()
            all_indices.extend(sampled)

        if len(all_indices) < total_samples:
            diff = total_samples - len(all_indices)
            all_indices.extend(all_indices[:diff])
        elif len(all_indices) > total_samples:
            all_indices = all_indices[:total_samples]

        all_indices = torch.tensor(all_indices)[torch.randperm(len(all_indices), generator=g)].tolist()

        #rank
        start = self.rank * self.num_samples_per_rank
        end = start + self.num_samples_per_rank
        return iter(all_indices[start:end])

    def __len__(self):
        return self.num_samples_per_rank
    
# Model Blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim > 0 else None
        self.short = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.conv1(self.act(self.norm1(x)))
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.short(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
    def forward(self, x, t):
        x = self.block1(x, t)
        x = self.block2(x, t)
        skip = x
        x = self.pool(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.block1 = ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
    def forward(self, x, skip, t):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t)
        x = self.block2(x, t)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4), nn.SiLU(),
            nn.Linear(dim*4, dim),
        )
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)

class UNet32(nn.Module):
    def __init__(self, base_ch=64, time_dim=128, in_channels=3):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down1 = Down(base_ch, base_ch*2, time_dim)
        self.down2 = Down(base_ch*2, base_ch*4, time_dim)
        self.mid1  = ResidualBlock(base_ch*4, base_ch*4, time_dim)
        self.mid2  = ResidualBlock(base_ch*4, base_ch*4, time_dim)
        self.up2   = Up(in_ch=base_ch*4, skip_ch=base_ch*4, out_ch=base_ch*2, time_emb_dim=time_dim)
        self.up1   = Up(in_ch=base_ch*2, skip_ch=base_ch*2, out_ch=base_ch,   time_emb_dim=time_dim)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(base_ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)
        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x

# LoRA
class LoRAConv2d(nn.Module):
    def __init__(self, base_conv: nn.Conv2d, r: int = 4, alpha: int = 16):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.A = nn.Conv2d(base_conv.in_channels, r, kernel_size=1, bias=False)
        self.B = nn.Conv2d(r, base_conv.out_channels, kernel_size=base_conv.kernel_size,
                           stride=base_conv.stride, padding=base_conv.padding,
                           dilation=base_conv.dilation, bias=False, groups=1)
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / float(r)
        self.register_buffer("lora_scale", torch.tensor(1.0))

    def forward(self, x):
        return self.base(x) + (self.lora_scale * self.scaling) * self.B(self.A(x))

def apply_lora_to_unet(model: nn.Module, r=4, alpha=16):
    for _, module in model.named_modules():
        for attr in ["conv1", "conv2", "in_conv", "out_conv", "pool", "upsample"]:
            if hasattr(module, attr):
                conv = getattr(module, attr)
                if isinstance(conv, nn.Conv2d):
                    setattr(module, attr, LoRAConv2d(conv, r=r, alpha=alpha))
    return model

def set_lora_scale(model, scale: float):
    for m in model.modules():
        if isinstance(m, LoRAConv2d):
            m.lora_scale.fill_(float(scale))

# Diffusion
def make_beta_schedule(num_steps=1000, schedule="linear", beta_start=1e-4, beta_end=0.02):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_steps)
    raise ValueError("only linear schedule implemented")

class DDPM(nn.Module):
    def __init__(self, model, image_size=32, channels=3, timesteps=1000, beta_schedule="linear"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = make_beta_schedule(timesteps, schedule=beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return a * x0 + b * noise
    
    def p_losses(self, x0, t, noise=None, reduction: str = "mean"):
        if noise is None:
            noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy, t.float())

        losses = F.mse_loss(noise_pred, noise, reduction="none")  # [B,C,H,W]
        losses = losses.mean(dim=(1, 2, 3))                       # [B]

        if reduction == "none":
            return losses
        elif reduction == "mean":
            return losses.mean()
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
        
    @torch.no_grad()
    def p_sample(self, x, t):
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(1.0 - betas_t)).clamp(max=3.0)
        model_mean = (x - betas_t * self.model(x, t.float()) / sqrt_one_minus_alphas_cumprod_t) * sqrt_recip_alphas_t
        if (t == 0).all():
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def sample(self, shape, device):
        self.model.eval()
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x.clamp(-1, 1)

# CIFAR
CIFAR10_LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def name_to_index(names):
    idx = []
    for n in names:
        n2 = n.strip().lower()
        if n2 in CIFAR10_LABELS:
            idx.append(CIFAR10_LABELS.index(n2))
        else:
            raise ValueError(f"Unknown class name: {n}")
    return sorted(set(idx))

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="./data", type=str)
    ap.add_argument("--epochs", default=200, type=int)
    ap.add_argument("--batch_size", default=128, type=int)
    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--lr", default=2e-4, type=float)
    ap.add_argument("--timesteps", default=1000, type=int)
    ap.add_argument("--save", default="checkpoints/ddpm_baseline.pt", type=str)
    ap.add_argument("--resume", default="", type=str)
    ap.add_argument("--lora_rank", default=0, type=int)
    ap.add_argument("--lora_alpha", default=16, type=int)
    ap.add_argument("--lora_scale", default=1.0, type=float)
    ap.add_argument("--sample_every", default=50, type=int)
    ap.add_argument("--classes", default="", type=str)
    ap.add_argument("--channels_last", action="store_true", help="use channels_last memory format")
    ap.add_argument("--class_weights_json", type=str, default=None)
    ap.add_argument("--weight_mode", type=str, default="sampling",
                choices=["sampling", "loss", "both"],)
    args = ap.parse_args()

    # DDP init
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    # optional class subset
    classes = None
    if args.classes.strip():
        classes = name_to_index(args.classes.split(","))
        if is_main_process():
            print(f"[Config] Using class subset: {args.classes} -> idx {classes}")

    # Build dataset
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_dataset = datasets.CIFAR10(root=args.dataset_root, train=True, download=True, transform=tf)
    if classes is not None and len(classes) > 0:
        idx_keep = [i for i, (_, y) in enumerate(train_dataset) if y in classes]
        train_dataset = Subset(train_dataset, idx_keep)
        if is_main_process():
            print(f"[Data] Filtered classes={classes} -> keep {len(idx_keep)} samples")

    # Choose sampler
    if args.class_weights_json:

        info = json.load(open(args.class_weights_json))
        classes = info.get("classes", CIFAR10_LABELS)
        cw = info["class_weights"]
        class_weight_dict = {i: float(cw[classes[i]]) for i in range(len(classes))}

        if args.weight_mode in ["sampling", "both"]:

            weights = _build_weight_tensors(train_dataset, args.class_weights_json)
            if is_dist():
                world = dist.get_world_size()
                steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * world))
                num_per_rank = steps_per_epoch * args.batch_size
#                sampler = WeightedDistributedSampler(
#                    weights, num_samples_per_rank=num_per_rank,
#                    replacement=True, seed=42
#                )
                sampler = QuotaDistributedSampler(
                    train_dataset, args.class_weights_json,
                    num_samples_per_rank=num_per_rank,
                    seed=42
                )
                shuffle = False
            else:
                sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
                shuffle = False
        else:
            if is_dist():
                sampler = DistributedSampler(train_dataset, shuffle=True)
                shuffle = False
            else:
                sampler, shuffle = None, True
    else:
        if is_dist():
            sampler = DistributedSampler(train_dataset, shuffle=True)
            shuffle = False
        else:
            sampler, shuffle = None, True

    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    #Build model
    unet = UNet32(base_ch=64)
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        unet.load_state_dict(sd["model"], strict=False)

    if args.lora_rank > 0:
        unet = apply_lora_to_unet(unet, r=args.lora_rank, alpha=args.lora_alpha)
        for n, p in unet.named_parameters():
            if "A.weight" in n or "B.weight" in n:
                p.requires_grad = True
            else:
                if p.requires_grad:
                    p.requires_grad = False

    set_lora_scale(unet, args.lora_scale)

    if args.channels_last:
        unet.to(memory_format=torch.channels_last)

    ddpm = DDPM(unet, timesteps=args.timesteps).to(device)

    if is_dist():
        ddpm = DDP(ddpm, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    params = [p for p in ddpm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    #Train
    for epoch in range(1, args.epochs+1):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        unwrap(ddpm).train()
    
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t = torch.randint(0, args.timesteps, (x.size(0),), device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=True):
                per_sample_loss = unwrap(ddpm).p_losses(x, t, reduction="none")  # [B]

                if args.class_weights_json and args.weight_mode in ["loss", "both"]:
                    weights = torch.tensor([class_weight_dict[int(lbl)] for lbl in y],
                                        device=per_sample_loss.device, dtype=per_sample_loss.dtype)
                    weights = weights / weights.mean()
                    # weighted_loss = (per_sample_loss * weights).mean()
                    weighted_loss = (per_sample_loss * weights).sum() / (weights.sum() + 1e-12)
                else:
                    weighted_loss = per_sample_loss.mean()

            scaler.scale(weighted_loss).backward()
            scaler.step(opt)
            scaler.update()

        # sampling + save
        if (epoch % args.sample_every == 0 or epoch == args.epochs) and is_main_process():
            core = unwrap(ddpm)
            with torch.no_grad():
                samples = core.sample((64,3,32,32), device)
                grid = vutils.make_grid((samples+1)/2, nrow=4)
                os.makedirs(os.path.dirname(args.save), exist_ok=True)
                vutils.save_image(grid, os.path.join(os.path.dirname(args.save), f"samples_e{epoch:03d}.png"))
            torch.save({"model": core.model.state_dict()}, args.save)

        if is_main_process():
            print(f"Epoch {epoch:03d} loss={weighted_loss.item():.4f}  (classes={args.classes or 'all'})")

    if is_dist():
        dist.destroy_process_group()
    if is_main_process():
        print(f"Saved {args.save}")

if __name__ == "__main__":
    main()