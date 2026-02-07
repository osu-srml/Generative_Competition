import math, os, torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
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

class LoRAConv2d(nn.Module):
    def __init__(self, base_conv: nn.Conv2d, r: int = 4, alpha: int = 16):
        super().__init__()
        self.base = base_conv
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Conv2d(base_conv.in_channels, r, 1, bias=False)
        self.B = nn.Conv2d(r, base_conv.out_channels, kernel_size=base_conv.kernel_size,
                           stride=base_conv.stride, padding=base_conv.padding,
                           dilation=base_conv.dilation, bias=False, groups=1)
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / float(r)
        self.register_buffer("lora_scale", torch.tensor(1.0))
    def forward(self, x):
        return self.base(x) + (self.lora_scale * self.scaling) * self.B(self.A(x))

def try_inject_lora(unet: nn.Module, r=4, alpha=16):
    for name, module in unet.named_modules():
        for attr in ["conv1", "conv2", "in_conv", "out_conv", "pool", "upsample"]:
            if hasattr(module, attr):
                conv = getattr(module, attr)
                if isinstance(conv, nn.Conv2d):
                    setattr(module, attr, LoRAConv2d(conv, r=r, alpha=alpha))
    return unet

def make_beta_schedule(num_steps=1000, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_steps)

class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = make_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
    @torch.no_grad()
    def p_sample(self, x, t):
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(1.0 - betas_t)).clamp(max=3.0)
        model_mean = (x - betas_t * self.model(x, t.float()) / sqrt_one_minus_alphas_cumprod_t) * sqrt_recip_alphas_t
        if (t == 0).all(): return model_mean
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(betas_t) * noise
    @torch.no_grad()
    def sample(self, n, device):
        self.model.eval()
        x = torch.randn((n,3,32,32), device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x.clamp(-1, 1)

def _infer_lora_rank_from_state(state):
    if isinstance(state, dict):
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and k.endswith("A.weight") and v.ndim == 4:
                return int(v.shape[0])  # r = out_channels of A
    return None

def load_generator(ckpt_path: str, device: torch.device, timesteps: int = 1000,
                   try_lora=True, lora_rank=None, lora_alpha=None, lora_scale=None):
    unet = UNet32(base_ch=64)
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    unet.load_state_dict(state, strict=False)

    has_lora = isinstance(state, dict) and any(k.endswith("A.weight") or k.endswith("B.weight") for k in state.keys())
    if try_lora and has_lora:
        r = lora_rank or _infer_lora_rank_from_state(state) or 4
        alpha = lora_alpha or max(16, 2*r)
        unet = try_inject_lora(unet, r=r, alpha=alpha)
        unet.load_state_dict(state, strict=False)
        if lora_scale is not None:
            for m in unet.modules():
                if isinstance(m, LoRAConv2d):
                    m.lora_scale.fill_(float(lora_scale))
        print(f"LoRA r={r}, alpha={alpha}, scale={lora_scale if lora_scale is not None else 1.0}")

    ddpm = DDPM(unet, timesteps=timesteps).to(device).eval()
    return ddpm
