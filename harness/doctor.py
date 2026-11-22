"""`agh doctor`: check that this server can run the harness (portability, 2026-11-22). Prints one line per check."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(name, fn):
    try:
        ok, msg = fn()
    except Exception as e:  # noqa: BLE001
        ok, msg = False, f"{type(e).__name__}: {e}"
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {msg}")
    return ok


def py_deps():
    import numpy, torch, yaml  # noqa: F401
    return True, f"python {sys.version.split()[0]} ({sys.executable}), torch {torch.__version__}, cuda {torch.version.cuda}"


def claude_cli():
    p = shutil.which("claude")
    if not p:
        return False, "`claude` not on PATH (install Claude Code and log in: `claude` once interactively)"
    v = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    return True, f"{p} {v}"


def agents():
    need = ["search-space", "explorer", "exploiter"]
    miss = [a for a in need if not (ROOT / ".claude" / "agents" / f"{a}.md").exists()]
    return not miss, "missing " + ", ".join(miss) if miss else "search-space, explorer, exploiter"


def gpus():
    from harness import gpu
    gs = gpu.list_gpus()
    if not gs:
        return False, "no GPU visible to nvidia-smi"
    want = int(os.environ.get("AGH_GPU", "0") or 0)
    rows = [f"{g.index}:{g.free_mb}MiB free,{g.util}% util" + (" <- AGH_GPU" if g.index == want else "") for g in gs]
    me = [g for g in gs if g.index == want]
    if not me:
        return False, f"AGH_GPU={want} not found; " + "; ".join(rows)
    others = gpu.other_users_on(me[0].uuid)
    warn = (f"  WARNING: GPU {want} has processes of other users {others}; pick a free card with AGH_GPU=<index>"
            if others else "")
    return True, "; ".join(rows) + warn


def sandbox():
    """bwrap with the project's mounts: run `import torch` inside, the same way candidates run."""
    from harness import sandbox as sb
    if not shutil.which("bwrap"):
        return False, "bwrap not installed (apt install bubblewrap)"
    scratch = ROOT / "runs" / ".doctor"
    scratch.mkdir(parents=True, exist_ok=True)
    argv = sb.build_argv([sys.executable, "-I", "-c", "import torch, numpy; print('sandbox ok')"],
                         mounts=sb.Mounts(ro={}, rw={str(scratch): "/out"}), cwd="/out", threads=1,
                         scratch=scratch / ".s")
    p = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    shutil.rmtree(scratch, ignore_errors=True)
    if p.returncode != 0:
        return False, (p.stderr or p.stdout).strip()[-400:] + "  (user namespaces may be disabled: " \
            "sysctl kernel.unprivileged_userns_clone / apparmor_restrict_unprivileged_userns)"
    return True, "bwrap runs python + torch with no network"


def private_dir():
    p = Path.home() / "ag_harness_private"
    p.mkdir(mode=0o700, exist_ok=True)
    mode = oct(p.stat().st_mode & 0o777)
    return mode == "0o700", f"{p} mode {mode} (gate verdicts, answer keys, GPU locks; agents cannot read it)"


def case_data():
    from harness import run_job
    rows, ok = [], True
    for c in sorted(d for d in (ROOT / "cases").iterdir() if (d / "budget.yaml").exists()):
        data = c / "data"
        if c.name.startswith("_"):
            continue
        if not data.exists():
            rows.append(f"{c.name}: no data/ (fine if your data_loader builds or fetches it; built-in cases: sync_data.sh)"); continue
        n = sum(1 for _ in data.iterdir())
        try:
            trainer = run_job.trainer_path(c).name
            has_tr = True
        except SystemExit:
            trainer, has_tr = run_job.trainer_of(c) if (c / "budget.yaml").exists() else "?", False
        rows.append(f"{c.name}: {n} files{' (symlink -> ' + str(data.resolve()) + ')' if data.is_symlink() else ''}"
                    f"{'' if has_tr else ', trainer ' + trainer + ' missing'}")
        ok &= has_tr
    return ok, "; ".join(rows)


def weave():
    if os.environ.get("AGH_WEAVE", "1") == "0":
        return True, "off (AGH_WEAVE=0)"
    wpy = os.environ.get("AGH_WEAVE_PY") or str(Path.home() / "miniforge3/bin/python")
    if not Path(wpy).exists():
        return True, f"optional, off: {wpy} not found (set AGH_WEAVE_PY to a python with `weave`)"
    p = subprocess.run([wpy, "-c", "import weave; print(weave.__version__)"], capture_output=True, text=True, timeout=120)
    return True, f"weave {p.stdout.strip()} via {wpy}" if p.returncode == 0 else f"optional, off: weave not importable in {wpy}"


def main():
    res = [check(n, f) for n, f in [("python deps", py_deps), ("claude CLI", claude_cli), ("agents", agents),
                                     ("GPU", gpus), ("sandbox", sandbox), ("private dir", private_dir),
                                     ("case data", case_data), ("weave logging", weave)]]
    print(json.dumps({"ready": all(res)}))
    return all(res)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
