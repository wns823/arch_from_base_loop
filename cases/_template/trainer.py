"""Generic trainer + scorer for a case (copied by `agh new-case`; you normally do NOT need to edit it).
Runs INSIDE the sandbox:  /case (ro) = your case folder,  /cand (ro) = the candidate (model.py, train_config.py),
/out (rw) = where result.json goes. No network, one GPU (or CPU with AGH_DEVICE=cpu).

It relies on four functions of your data_loader.py (fixed code that agents cannot edit):
  load(split)                 split in {train, val, test} -> (xs, y): xs = tuple of tensors with the same first dim
  model_args(xs, conditional) -> tuple of positional args for model(...) (drop the condition when not conditional)
  score(pred, y)              -> float, the selection score (set higher_is_better in budget.yaml)
  example_inputs(conditional) -> args for ONE sample (size check); optional model_init_kwargs() -> dict for Model()
Training: random minibatches of train_config.BATCH_SIZE until --train_samples samples were seen,
loss = train_config.loss_fn(pred, y), optimizer = make_optimizer(params), optional make_scheduler(opt, total_steps).
Evaluation: EVAL_BATCH samples per call (default 1 = each sample on its own, so a model cannot use statistics
across evaluated samples). Writes /out/result.json with score, train_time_s and a few diagnostics.
CLI contract with the harness: --train_samples N --eval val|test --seed S --model_kwargs JSON (+ final_extra_args)."""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="/case"); ap.add_argument("--cand", default="/cand"); ap.add_argument("--out", default="/out")
    ap.add_argument("--train_samples", type=int, required=True)
    ap.add_argument("--eval", choices=["val", "test"], default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_kwargs", default="{}")
    a, _extra = ap.parse_known_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, a.case)
    dl = load_module(os.path.join(a.case, "data_loader.py"), "data_loader")
    mm = load_module(os.path.join(a.cand, "model.py"), "model")
    cfg = load_module(os.path.join(a.cand, "train_config.py"), "train_config")
    cond = bool(getattr(mm, "CONDITIONAL", False))
    kw = dl.model_init_kwargs() if hasattr(dl, "model_init_kwargs") else {}
    model = mm.Model(**kw, **json.loads(a.model_kwargs)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    xs, y = dl.load("train")
    xs, y = tuple(x.to(dev) for x in xs), y.to(dev)
    n = y.shape[0]
    opt = cfg.make_optimizer(model.parameters())
    bs = int(cfg.BATCH_SIZE)
    total_steps = max(1, (a.train_samples + bs - 1) // bs)
    sched = cfg.make_scheduler(opt, total_steps) if hasattr(cfg, "make_scheduler") else None
    model.train(); t0 = time.time(); seen = step = 0; loss = torch.tensor(float("nan"))
    while seen < a.train_samples:
        i = torch.randint(0, n, (bs,), device=dev)
        pred = model(*dl.model_args(tuple(x[i] for x in xs), cond))
        loss = cfg.loss_fn(pred, y[i])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        seen += bs; step += 1
        if not torch.isfinite(loss):
            break
    train_time = time.time() - t0

    xe, ye = dl.load(a.eval)
    eb = int(getattr(dl, "EVAL_BATCH", 1))
    model.eval(); preds = []
    with torch.no_grad():
        for j in range(0, ye.shape[0], eb):
            preds.append(model(*dl.model_args(tuple(x[j:j + eb].to(dev) for x in xe), cond)).float().cpu())
    score = float(dl.score(torch.cat(preds), ye))
    res = {"score": score, "eval": a.eval, "seed": a.seed, "train_samples": a.train_samples, "steps": step,
           "final_loss": float(loss), "params": n_params, "conditional": cond, "train_time_s": train_time,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
