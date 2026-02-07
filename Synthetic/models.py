import json, numpy as np
from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class RBFModel:
    name: str
    bias: float
    centers: np.ndarray
    amps: np.ndarray
    sigmas: np.ndarray
    def score(self, thetas: np.ndarray) -> np.ndarray:
        val = np.zeros((thetas.shape[0],), dtype=float) + float(self.bias)
        for m, a, s in zip(self.centers, self.amps, self.sigmas):
            diff = thetas - m
            val += float(a) * np.exp(- np.sum(diff**2, axis=1) / (2*float(s)**2))
        return np.clip(val, 0.0, 1.0)

def load_models_from_json(path: str) -> List[RBFModel]:
    data = json.load(open(path, "r", encoding="utf-8"))
    models = []
    for spec in data["models"]:
        models.append(RBFModel(
            name   = spec["name"],
            bias   = float(spec.get("bias", 0.0)),
            centers= np.array(spec.get("centers", [[0,0]]), dtype=float),
            amps   = np.array(spec.get("amps", [1.0]), dtype=float),
            sigmas = np.array(spec.get("sigmas", [1.0]), dtype=float),
        ))
    return models

def save_models_csv(path, models):
    rows = []
    for m in models:
        rows.append({
            "name": m.name, "bias": m.bias, "R": len(m.amps),
            "centers": ";".join([f"[{c[0]:.3f},{c[1]:.3f}]" for c in m.centers]),
            "amps": ";".join([f"{a:.3f}" for a in m.amps]),
            "sigmas": ";".join([f"{s:.3f}" for s in m.sigmas]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
