"""Fixed data access for this case (agents read it but cannot edit it). REPLACE the toy data with yours.

Toy example (runs out of the box): one-hot sequences seq [n, 4, 64] and a context vector ctx [n, 8]; the response y
depends on motif counts with context-dependent gains. Val/test contexts come from held-out clusters, so a model
that ignores ctx cannot do well. Data is generated on first use and cached in data/.

Contract with trainer.py and the harness (keep these names):
  load(split) -> (xs, y)              xs: tuple of tensors, same first dim as y
  model_args(xs, conditional) -> args for model(...)
  score(pred, y) -> float             selection score; higher is better unless budget.yaml says higher_is_better: false
  example_inputs(conditional) -> args for ONE sample (size check)
  model_init_kwargs() -> dict         constructor kwargs for Model() (optional)
  EVAL_BATCH = 1                      samples per model call at evaluation (1 = no cross-sample statistics)
"""
import os

import numpy as np
import torch

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
L, CTX = 64, 8
N = {"train": 8000, "val": 2000, "test": 2000}
EVAL_BATCH = 1


def _make(split):
    rng = np.random.default_rng(0)                      # shared world: motifs and gain map
    motifs = rng.integers(0, 4, (4, 6))
    W = rng.normal(size=(CTX, 4))
    centers = rng.normal(size=(60, CTX)) * 1.5          # 40 train clusters, 10 val, 10 test (enough for val ≈ test)
    cl = {"train": range(0, 40), "val": range(40, 50), "test": range(50, 60)}[split]
    r = np.random.default_rng({"train": 1, "val": 2, "test": 3}[split])
    n = N[split]
    seq = r.integers(0, 4, (n, L))
    for m in range(4):
        k = r.integers(0, 3, n)
        for i in np.nonzero(k)[0]:
            for _ in range(k[i]):
                p = r.integers(0, L - 6); seq[i, p:p + 6] = motifs[m]
    win = np.lib.stride_tricks.sliding_window_view(seq, 6, axis=1)          # [n, L-5, 6]
    cnt = np.stack([(win == motifs[m]).all(-1).sum(1) for m in range(4)], 1).astype(np.float32)
    ctx = centers[r.choice(list(cl), n)] + 0.3 * r.normal(size=(n, CTX))
    y = (cnt * np.tanh(ctx @ W)).sum(1) + 0.1 * r.normal(size=n)
    return seq, ctx.astype(np.float32), y.astype(np.float32)


def load(split):
    f = os.path.join(D, f"{split}.npz")
    if os.path.exists(f):
        z = np.load(f); seq, ctx, y = z["seq"], z["ctx"], z["y"]
    else:
        seq, ctx, y = _make(split)
        try:
            os.makedirs(D, exist_ok=True); np.savez(f, seq=seq, ctx=ctx, y=y)
        except OSError:
            pass   # read-only inside the sandbox: run `python cases/<name>/data_loader.py` once on the host
    oh = np.eye(4, dtype=np.float32)[seq].transpose(0, 2, 1)
    return (torch.from_numpy(np.ascontiguousarray(oh)), torch.from_numpy(ctx)), torch.from_numpy(y)


def model_args(xs, conditional):
    seq, ctx = xs
    return (seq, ctx if conditional else None)


def score(pred, y):
    """R² (higher is better)."""
    p, t = pred.double().reshape(-1), y.double().reshape(-1)
    return float(1 - ((t - p) ** 2).mean() / ((t - t.mean()) ** 2).mean())


def example_inputs(conditional):
    x = torch.zeros(1, 4, L); x[:, 0] = 1
    return (x, torch.zeros(1, CTX) if conditional else None)


def model_init_kwargs():
    return {"ctx_dim": CTX}


if __name__ == "__main__":   # host side: materialise data/ once
    for s in N:
        load(s)
        print(s, "cached in", D)
