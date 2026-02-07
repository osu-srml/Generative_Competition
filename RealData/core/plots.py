import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def per_round_last(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("round").tail(1).copy()

def plot_steps(df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(10,4))
    sub = df[df["actor"]>=0]
    for i in sorted(sub["player"].unique()):
        s = sub[sub["player"]==i]
        plt.plot(s["step"], s["U"], marker='o', label=f"P{i}")
    plt.xlabel("Step")
    plt.ylabel("U")
    plt.title("Best-Response actor utilities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def heatmap_U(df: pd.DataFrame, out_png: str, P: int):
    eval_rows = df[df["actor"] == -1].copy()
    if eval_rows.empty:
        raise ValueError("No rows. ")

    rounds = sorted(eval_rows["round"].unique())
    U = np.full((P, len(rounds)), np.nan, dtype=float)

    for j, r in enumerate(rounds):
        rr = eval_rows[eval_rows["round"] == r]
        for i in range(P):
            vals = rr[rr["player"] == i]["U"].values
            if vals.size > 0:
                U[i, j] = float(vals[-1])

    fig, ax = plt.subplots(figsize=(1.2*len(rounds)+2, 0.8*P+2))
    im = ax.imshow(U, aspect='auto', cmap='viridis')
    ax.set_xticks(np.arange(len(rounds)))
    ax.set_xticklabels([str(r) for r in rounds])
    ax.set_yticks(np.arange(P))
    ax.set_yticklabels([f"P{i}" for i in range(P)])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="U")
    ax.set_title("U heatmap (Players × Rounds)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def heatmap_WD(df: pd.DataFrame, out_png: str):
    per = per_round_last(df)
    rounds = sorted(per["round"].unique())
    M = np.vstack([per["W"].to_numpy(),
                   per["shannon"].to_numpy(),
                   (1.0 - per["hhi"].to_numpy())])
    fig, ax = plt.subplots(figsize=(1.2*len(rounds)+2, 3.2))
    im = ax.imshow(M, aspect='auto', cmap='viridis')
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(["Welfare","Shannon","1-HHI"])
    ax.set_xticks(np.arange(len(rounds)))
    ax.set_xticklabels([str(r) for r in rounds])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Value")
    ax.set_title("W / Diversity (per Round)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def coverage_heatmap(S, pi, chosen, model_names, user_names, out_png: str):
    K, M = S.shape
    sub = S[:, chosen]
    winners = sub.argmax(axis=1)
    Pn = len(chosen)
    cov = np.zeros((Pn, K), dtype=float)
    for k in range(K):
        cov[winners[k], k] = float(pi[k])
    fig, ax = plt.subplots(figsize=(1.2*K+2, 0.8*Pn+2))
    im = ax.imshow(cov, aspect='auto', cmap='viridis')
    ax.set_yticks(np.arange(Pn))
    ax.set_yticklabels([f"P{i}({model_names[m]})" for i,m in enumerate(chosen)])
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(user_names, rotation=45, ha='right', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mass")
    ax.set_title("Coverage heatmap")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()