"""Host-side runner: lease the GPU, run one training/eval job in the bwrap sandbox, return result.json.

Usage (with an interpreter whose venv has torch):
  python -m harness.run_job --case cases/V0 --cand <dir with model.py, train_config.py> --out <dir>
         --train_samples N [--eval val|test] [--seed S] [--model_kwargs JSON] [--min_free_mb 4000]
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from harness import gpu as gpumod          # noqa: E402
from harness import sandbox                 # noqa: E402


def trainer_of(case):
    """Fixed trainer/scorer of a case (2026-11-22: no silent default to another case's trainer):
    1. `trainer:` in budget.yaml, a file name in harness/ (train_v2.py) or in the case folder;
    2. else cases/<case>/trainer.py.
    Returns the file name (resolved by trainer_path)."""
    import yaml
    case = Path(case)
    b = case / "budget.yaml"
    t = (yaml.safe_load(open(b)) or {}).get("trainer") if b.exists() else None
    if t:
        return t
    if (case / "trainer.py").exists():
        return "trainer.py"
    raise SystemExit(f"{case.name}: no trainer. Put the trainer/scorer in cases/{case.name}/trainer.py "
                     "(template: cases/_template/trainer.py) or set `trainer:` in budget.yaml")


def trainer_path(case):
    """Host path of the case's trainer: the case folder wins over harness/."""
    case = Path(case)
    t = trainer_of(case)
    for p in (case / t, HERE / t):
        if p.exists():
            return p.resolve()
    raise SystemExit(f"trainer {t} not found in cases/{case.name}/ or harness/")


def extra_ro(case):
    """Host dirs bound read-only into the sandbox at the same path:
    * budget.yaml `extra_ro_mounts` (each entry may use ~ and $VARS), and
    * the target of `cases/<case>/data` when it is a symlink to a directory outside the repo (e.g. V3's 81 GB
      embedding cache), so a case's data can live on any disk of any server (portability fix 2026-11-22)."""
    import yaml
    case = Path(case)
    b = case / "budget.yaml"
    dirs = (yaml.safe_load(open(b)) or {}).get("extra_ro_mounts", []) if b.exists() else []
    out = {}
    for d in dirs or []:
        p = Path(os.path.expandvars(os.path.expanduser(str(d)))).resolve()
        if p.exists():
            out[str(p)] = str(p)
    data = case / "data"
    if data.is_symlink():
        t = data.resolve()
        if t.is_dir():
            out[str(t)] = str(t)
    return out


def run(case, cand, out, train_samples, eval_split="val", seed=0, model_kwargs="{}", min_free_mb=4000,
        timeout_s=None, extra_args=(), trainer=None, loader=None):
    """extra_args: further trainer flags, e.g. ["--train_cells", "trainval"] (V2 final mode, 2026-11-20).
    trainer / loader (2026-11-22): the run's frozen copies (runs/<case>/<run>/frozen/), so a run keeps the exact
    trainer and data loader it started with even if harness/ or the case is edited later. Default: the live files.
    AGH_DEVICE=cpu: no GPU lease, the job runs on CPU in the same sandbox (smoke tests, CPU-only servers)."""
    case, cand, out = Path(case).resolve(), Path(cand).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / ".scratch"
    tpath = Path(trainer).resolve() if trainer else trainer_path(case)
    tname = tpath.name
    ro = {str(case): "/case"}
    if loader:
        ro[str(Path(loader).resolve())] = "/case/data_loader.py"   # bound over the live loader (later binds win)
    ro.update({str(cand): "/cand", str(tpath): f"/harness/{tname}", **extra_ro(case)})
    if (HERE / "fid_inception.py").exists():
        ro[str(HERE / "fid_inception.py")] = "/harness/fid_inception.py"
    mounts = sandbox.Mounts(ro=ro, rw={str(out): "/out"})
    argv_cmd = [sys.executable, "-I", f"/harness/{tname}", "--train_samples", str(train_samples),
                "--eval", eval_split, "--seed", str(seed), "--model_kwargs", model_kwargs, *map(str, extra_args)]
    if os.environ.get("AGH_DEVICE", "").lower() == "cpu":
        lease = contextlib.nullcontext(None)
    else:
        lease = gpumod.lease(min_free_mb=min_free_mb, wait_s=6 * 3600, poll_s=60, case=case.name)
    with lease as g:
        argv = sandbox.build_argv(argv_cmd, mounts=mounts, gpu_device=g.device if g else None,
                                  gpu_uuid=g.uuid if g else None, cwd="/out", threads=4, scratch=scratch)
        log = open(out / "job.log", "w")
        log.write(f"# gpu index={g.index} uuid={g.uuid} device={g.device}\n" if g else "# device=cpu (AGH_DEVICE=cpu)\n")
        log.flush()
        p = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    subprocess.run(["rm", "-rf", str(scratch)])
    res_path = out / "result.json"
    if p.returncode != 0 or not res_path.exists():
        return {"failed": True, "returncode": p.returncode, "log": str(out / "job.log")}
    return json.load(open(res_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_samples", type=int, required=True)
    ap.add_argument("--eval", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_kwargs", default="{}")
    ap.add_argument("--min_free_mb", type=int, default=4000)
    a = ap.parse_args()
    print(json.dumps(run(a.case, a.cand, a.out, a.train_samples, a.eval, a.seed, a.model_kwargs, a.min_free_mb)))
