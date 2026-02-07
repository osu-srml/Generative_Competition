import json, torch
import torch.nn.functional as F
from typing import Dict, List

CIFAR10_LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
LABEL_TO_IDX = {n:i for i,n in enumerate(CIFAR10_LABELS)}

def load_reward_model(hub_entry: str = "cifar10_resnet20", pretrained: bool = True, device=None):
    #Load pretrained CIFAR-10/100 classifier from chenyaofo/pytorch-cifar-models via torch.hub.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_entry, pretrained=pretrained)  # downloads if needed
    model.to(device).eval()
    return model

def normalize_for_cifar(x: torch.Tensor) -> torch.Tensor:
    # standard CIFAR-10 normalization
    mean = x.new_tensor([0.4914, 0.4822, 0.4465])[None, :, None, None]
    std  = x.new_tensor([0.2023, 0.1994, 0.2010])[None, :, None, None]
    return (x - mean) / std

@torch.no_grad()
def reward_scores(imgs_in_minus1_1: torch.Tensor, model, group_weights: Dict[str, float], mode: str = "probs") -> torch.Tensor:
    x = (imgs_in_minus1_1 + 1.0) / 2.0  # back to [0,1]
    x = normalize_for_cifar(x)
    logits = model(x)
    probs = logits.softmax(dim=1)
    w = torch.zeros(probs.size(1), device=probs.device)
    for name, val in group_weights.items():
        w[LABEL_TO_IDX[name]] = float(val)
    if mode == "probs":
        r = (probs * w[None]).sum(dim=1)
    elif mode == "acc":
        pred = probs.argmax(dim=1)
        r = w[pred]
    else:
        raise ValueError("mode must be 'probs' or 'acc'")
    return r
