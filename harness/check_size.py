"""Sandbox-side size check (CPU): build the candidate model and count parameters and forward FLOPs for ONE sample.

Mounts: /case (ro), /cand (ro), /out (rw). Writes /out/size.json. Never runs on the host (LLM code).
"""
import importlib.util
import json
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    case, cand, out = "/case", "/cand", "/out"
    sys.path.insert(0, case)
    dl = load(os.path.join(case, "data_loader.py"), "data_loader")
    mm = load(os.path.join(cand, "model.py"), "model")
    conditional = bool(getattr(mm, "CONDITIONAL", False))
    kw = dl.model_init_kwargs() if hasattr(dl, "model_init_kwargs") else {}
    model = mm.Model(**kw).eval()
    n_params = sum(p.numel() for p in model.parameters())
    args = dl.example_inputs(conditional)          # one sample, case-defined (fixed code)
    with FlopCounterMode(display=False) as fc:   # grad enabled: no_grad views of parameters break the module tracker
        model(*args)
    res = {"params": int(n_params), "flops": int(fc.get_total_flops()), "conditional": conditional}
    json.dump(res, open(os.path.join(out, "size.json"), "w"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
