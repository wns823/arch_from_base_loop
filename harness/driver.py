"""Run driver (REQ-016). Code, not an agent, decides what runs next, calls agents, trains and scores candidates.

Run directory runs/<case>/<run>/:
  workspace/        task.md, base/, data_loader.py (read-only reference for agents)
  boundary.json     fixed outer boundary (editable files, budget, size limits, date cap, run limits)
  space_v<k>.json   search spaces written by the SSA; registered ones are listed in state.json
  cands/<id>/       model.py, train_config.py, notes.json (agent) + result.json (loop, after training)
  results/<id>/     sandbox outputs (job.log, size.json, result.json) — loop only
  LEDGER.md         regenerated after every event; the agents' only memory
  state.json        loop state (bandit, SSA history, finalize lock)
  events.jsonl      one line per loop event; tokens.jsonl one line per agent call (usage, cost)
  search_log.jsonl  SSA web calls (PostToolUse hook)
"""
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import sys
from pathlib import Path

import yaml

from harness import sandbox
from harness import run_job

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "harness"
EDITABLE = ["model.py", "train_config.py"]
CAND_ALLOWED = set(EDITABLE) | {"notes.json", "result.json", "__pycache__"}

# allocator constants (project_spec §6.3)
K_STAGNATION = 6      # candidates without improving best by DELTA·σ → SSA
DELTA = 0.5
COVERAGE_C = 0.7      # coverage of the current space → SSA
MAX_INTERVAL_M = 12   # candidates since the last SSA → SSA
STOP_SSA_ROUNDS = 2   # SSA rounds in a row without improvement → stop
MIN_LEFT_FOR_SSA = 8  # no SSA revision with fewer candidates left (fix 2026-11-22)
MAX_SSA_REVISIONS = 5 # outer loop cap: revisions after the first space (user decision 2026-11-20)
TARGET_WIDEN_SIGMA = 3.0  # revision widens the space when the best val is more than this many σ below the target
P_OFF_AXIS = 0.2      # explorer: chance of an off-axis edit
UCB_C = math.sqrt(2)
REWARD_CLIP = 3.0      # delta reward clip, in seed σ
REWARD_WINDOW = 8      # delta reward: mean over the agent's last K rewards
ARM_WINDOW, ARM_MIN = 8, 2   # each agent makes at least ARM_MIN of the last ARM_WINDOW candidates (fix 2026-11-22)
MAX_RETRIES = 2


# ---------------------------------------------------------------- helpers
def rdir(case, run):
    return PROJECT / "runs" / case / run


def jload(p, default=None):
    p = Path(p)
    return json.load(open(p)) if p.exists() else default


def jsave(p, obj):
    tmp = Path(str(p) + ".tmp")
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def event(R, kind, **kw):
    with open(R / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **kw}) + "\n")


def cand_ids(R):
    return sorted(p.name for p in (R / "cands").iterdir() if p.is_dir())


def scored(R):
    """[(id, S_val, notes)] for successfully trained candidates."""
    out = []
    for cid in cand_ids(R):
        res = jload(R / "cands" / cid / "result.json", {})
        if res.get("score") is not None and not res.get("failed"):
            out.append((cid, res["score"], jload(R / "cands" / cid / "notes.json", {})))
    return out


def latest_space(R, st):
    if not st["spaces"]:
        return None, None
    k = st["spaces"][-1]
    return k, jload(R / f"space_v{k}.json")


def eff_axes(R, space, cache=None):
    """{cid: {axis: choice}} — the module choices a candidate actually has (review fix 2026-11-20).
    000 has the space's `base`; every other candidate inherits its parent's choices and overrides them with its
    own notes['axes'] (agents may record only what they changed)."""
    cache = {} if cache is None else cache
    base = dict((space or {}).get("base") or {})

    def get(cid, depth=0):
        if cid in cache:
            return cache[cid]
        n = jload(R / "cands" / cid / "notes.json", {}) or {}
        p = n.get("parent")
        if cid == "000" or not p or depth > 200 or not (R / "cands" / p).exists():
            ax = dict(base)
        else:
            ax = dict(get(p, depth + 1))
        if cid != "000":
            ax.update({a: c for a, c in (n.get("axes") or {}).items() if isinstance(c, str)})
        cache[cid] = ax
        return ax

    return {cid: get(cid) for cid in cand_ids(R)}


HP_SKIP = re.compile(r"^(N_TRAIN|CONDITIONAL$)")


def hp_of(d):
    """Hyperparameters read from the candidate's code, not from what the agent wrote (review fix 2026-11-20):
    module-level UPPER_CASE literals in train_config.py / model.py, the optimizer and scheduler classes used, and the
    loss functions called in loss_fn."""
    import ast
    out = {}
    for f in ("train_config.py", "model.py"):
        p = Path(d) / f
        if not p.exists():
            continue
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.isupper() and not HP_SKIP.match(name):
                    try:
                        out[name] = ast.literal_eval(node.value)
                    except Exception:  # noqa: BLE001
                        pass
            if f == "train_config.py" and isinstance(node, ast.FunctionDef):
                src = ast.get_source_segment(p.read_text(), node) or ""
                if node.name == "make_optimizer":
                    for call in ast.walk(node):
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) \
                                and "optim" in (ast.get_source_segment(p.read_text(), call.func) or ""):
                            out["optimizer"] = call.func.attr
                            for kw in call.keywords:
                                if kw.arg in ("lr", "weight_decay") and isinstance(kw.value, ast.Name):
                                    continue   # refers to a constant already listed (LR, WEIGHT_DECAY)
                                if kw.arg and kw.arg != "params":
                                    out[kw.arg] = ast.get_source_segment(p.read_text(), kw.value)
                            break
                elif node.name == "make_scheduler":
                    m = re.findall(r"lr_scheduler\.(\w+)\(", src)
                    out["scheduler"] = m[0] if m else "custom"
                elif node.name == "loss_fn":
                    m = sorted(set(re.findall(r"(?:F|functional)\.(\w+)\(", src)))
                    if m:
                        out["loss"] = "+".join(m)
    out.setdefault("scheduler", "none")
    return out


def lineage_ok(R, cid, root):
    """True if cid descends from root through exploiter candidates only (root itself counts)."""
    seen = 0
    while cid and seen < 200:
        if cid == root:
            return True
        n = jload(R / "cands" / cid / "notes.json", {}) or {}
        if n.get("agent") != "exploiter":
            return False
        cid, seen = n.get("parent"), seen + 1
    return False


# ---------------------------------------------------------------- init
BUDGET_KEYS = {
    # key: (required?, default, meaning)  — `agh new-case` writes a commented budget.yaml with all of these
    "train_samples": (True, None, "training samples seen per candidate (the search budget, unit set by your trainer)"),
    "candidates_per_run": (False, 30, "maximum candidates per run"),
    "mode": (False, None, "reproduce (paper target, date-capped web search, leak check) or improve (your own target)"),
    "trainer": (False, None, "trainer/scorer file; default cases/<case>/trainer.py"),
    "target": (False, None, "{value, metric, higher_is_better}; shown to agents as a number only"),
    "higher_is_better": (False, None, "direction of the trainer's score (default: target.higher_is_better, else true)"),
    "sigma_seed": (False, "auto", "run-to-run s.d. of the score; auto = train the base model with seeds 0,1,2 at init"),
    "max_params": (False, None, "parameter limit; default size_factor x base model"),
    "max_flops": (False, None, "forward FLOPs per example_inputs() call limit; default size_factor x base model"),
    "size_factor": (False, 4.0, "limit = this x base model when max_params / max_flops are not given"),
    "h_train_time_s": (False, None, "reference train time; default = the base model's measured train time"),
    "timeout_ratio_to_H_time": (False, 3.0, "candidate timeout = this x h_train_time_s + 300 s"),
    "search_date_cap": (False, None, "reproduce mode only: the SSA may cite only sources dated before this"),
    "ssa_protocol": (False, 2, "2 = module axes + hyperparameter ranges (explorer swaps modules, exploiter tunes)"),
    "reward_mode": (False, "delta", "bandit reward: delta = (score - best) / sigma, clipped"),
    "notes_for_agents": (False, None, "list of constraints/ideas shown to every agent (e.g. 'keep the encoder')"),
    "final_train_samples": (False, None, "retrain the selected model with this budget before the one test score"),
    "final_extra_args": (False, None, "extra trainer flags for the final training"),
    "reselect_top3": (False, False, "retrain the top 3 with seed 1 on val before picking the final model"),
}


def load_budget(case):
    """Read cases/<case>/budget.yaml, check it and fill defaults (2026-11-22: clear errors instead of KeyError).
    Returns (budget, mode). Existing reproduction cases (with search_date_cap) keep their exact behaviour."""
    C = PROJECT / "cases" / case
    if not (C / "budget.yaml").exists():
        raise SystemExit(f"cases/{case}/budget.yaml not found (create a case with: harness/bin/agh new-case {case})")
    b = yaml.safe_load(open(C / "budget.yaml")) or {}
    mode = b.get("mode") or ("reproduce" if b.get("search_date_cap") else "improve")
    errs = [f"  {k}: {m}" for k, (req, _, m) in BUDGET_KEYS.items() if req and b.get(k) is None]
    if mode not in ("reproduce", "improve"):
        errs.append(f"  mode: must be reproduce or improve (got {mode})")
    if mode == "reproduce":
        for k in ("search_date_cap", "max_params", "max_flops", "h_train_time_s"):
            if b.get(k) is None:
                errs.append(f"  {k}: required in reproduce mode ({BUDGET_KEYS[k][2]}); run the case gate")
        if b.get("sigma_seed") in (None, "auto"):
            errs.append("  sigma_seed: required in reproduce mode (measured by the case gate)")
    tg = b.get("target")
    if tg is not None and (not isinstance(tg, dict) or "value" not in tg):
        errs.append("  target: must be {value: <number>, metric: <text>, higher_is_better: true|false}")
    sg = b.get("sigma_seed", "auto")
    if sg != "auto" and not (isinstance(sg, (int, float)) and sg > 0):
        errs.append("  sigma_seed: a positive number or auto")
    if errs:
        raise SystemExit(f"cases/{case}/budget.yaml is not valid:\n" + "\n".join(errs))
    for k, (_, d, _) in BUDGET_KEYS.items():
        b.setdefault(k, d)
    if b["higher_is_better"] is None:
        b["higher_is_better"] = bool((tg or {}).get("higher_is_better", True))
    if tg is not None:
        tg.setdefault("metric", "the selection score")
        tg["higher_is_better"] = bool(b["higher_is_better"])
    run_job.trainer_path(C)   # fails early with a clear message when the trainer is missing
    return b, mode


def internal(B, x):
    """Scores are maximised everywhere in the loop; a lower-is-better metric is stored negated (2026-11-22)."""
    return x if (x is None or B.get("higher_is_better", True)) else -x


def init_run(case, run=None):
    C = PROJECT / "cases" / case
    budget, mode = load_budget(case)
    run = run or time.strftime("r%Y%m%d-%H%M%S")
    R = rdir(case, run)
    if R.exists():
        raise SystemExit(f"{R} exists")
    (R / "workspace").mkdir(parents=True)
    shutil.copy(C / "task.md", R / "workspace" / "task.md")
    # target performance (user decision 2026-11-20): agents get base + task + the target number, never the source
    tgt = budget.get("target")
    with open(R / "workspace" / "task.md", "a") as fh:
        if tgt and mode == "reproduce":
            fh.write(f"\n## Target performance\n\nA published method reaches **{tgt['value']}** on this task "
                     f"({tgt['metric']}). Aim to reach or exceed it within the fixed budget. "
                     "The source is not given and must not be searched for.\n")
        elif tgt:
            fh.write(f"\n## Target performance\n\nThe user's target is **{tgt['value']}** ({tgt['metric']}; "
                     f"{'higher' if tgt['higher_is_better'] else 'lower'} is better). Aim to reach it within the fixed budget.\n")
        if budget.get("notes_for_agents"):
            fh.write("\n## Constraints and ideas from the user\n\n" + "".join(f"- {x}\n" for x in budget["notes_for_agents"]))
    shutil.copy(C / "data_loader.py", R / "workspace" / "data_loader.py")
    shutil.copytree(C / "base", R / "workspace" / "base", ignore=shutil.ignore_patterns("__pycache__"))
    # frozen copies (user request 2026-11-22): the run keeps the exact trainer, loader, budget and task it started
    # with; every job of the run uses frozen/<trainer> and frozen/data_loader.py
    import hashlib
    fz = R / "frozen"
    fz.mkdir()
    tpath = run_job.trainer_path(C)
    for src in (tpath, C / "data_loader.py", C / "budget.yaml", R / "workspace" / "task.md"):
        shutil.copy(src, fz / src.name)
    jsave(fz / "MANIFEST.json", {"trainer": tpath.name, "trainer_source": str(tpath.relative_to(PROJECT)),
                                 "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                            for f in sorted(fz.iterdir()) if f.name != "MANIFEST.json"}})
    boundary = {
        "case": case, "mode": mode, "editable_files": EDITABLE,
        "frozen": ["data_loader.py", "data/", f"scorer (frozen/{tpath.name})", "split", "budget"],
        "trainer_path": str(fz / tpath.name), "loader_path": str(fz / "data_loader.py"),
        "train_samples": budget["train_samples"],
        "max_params": budget["max_params"], "max_flops_per_sample": budget["max_flops"],
        "timeout_s": (int(budget["timeout_ratio_to_H_time"] * budget["h_train_time_s"]) + 300
                      if budget["h_train_time_s"] else None),
        "search_date_cap": str(budget["search_date_cap"]) if budget["search_date_cap"] else None,
        "candidates_per_run": budget["candidates_per_run"],
        "selection_score": "S_val (see task.md)",
        "target": tgt, "higher_is_better": bool(budget["higher_is_better"]),
        "ssa_protocol": int(budget["ssa_protocol"]),
        "reselect_top3": bool(budget["reselect_top3"]),
    }
    # final training (paper fidelity, 2026-11-20): the selected model is retrained once at the paper's length
    # (and, if given, on train+val units) and evaluated on test. Absent keys = same budget as the search.
    if budget.get("final_train_samples"):
        extra = list(budget.get("final_extra_args") or [])
        if budget.get("final_train_cells"):
            extra += ["--train_cells", str(budget["final_train_cells"])]
        ref = budget.get("final_h_train_time_s") or budget["h_train_time_s"]
        boundary["final"] = {"train_samples": int(budget["final_train_samples"]), "extra_args": extra,
                             "timeout_s": int(budget["timeout_ratio_to_H_time"] * ref) + 300 if ref else None}
    jsave(R / "boundary.json", boundary)
    sigma0 = budget["sigma_seed"] if budget["sigma_seed"] != "auto" else 1.0   # placeholder until measured below
    # reward_margin_sigma (budget.yaml, default 0): a candidate counts as "new best" / "beat parent" for the bandit
    # only if it beats the reference by this many seed σ (user decision 2026-11-19, from V6 on)
    jsave(R / "state.json", {"case": case, "run": run, "sigma": sigma0, "spaces": [],
                             "reward_margin": float(budget.get("reward_margin_sigma", 0)) * sigma0,
                             "policy": os.environ.get("AGH_POLICY", "ucb"),
                             "reward_mode": budget["reward_mode"],
                             # bandit fix 2026-11-22: asymmetric clip so one bad untuned module cannot starve an agent
                             "reward_clip": [-1.0, REWARD_CLIP], "reward_fail": -2.0,
                             "ssa_log": [], "bandit": {"exploiter": [0, 0.0], "explorer": [0, 0.0]},
                             "since_ssa": 0, "finalized": False, "next_id": 1})
    (R / "cands" / "000").mkdir(parents=True)
    for f in EDITABLE:
        shutil.copy(C / "base" / f, R / "cands" / "000" / f)
    jsave(R / "cands" / "000" / "notes.json", {"agent": "base", "parent": None, "axes": {}, "idea": "base model A0"})
    event(R, "init", boundary=boundary)
    # improve mode without size limits: measure the base model, limits = size_factor x base
    if boundary["max_params"] is None or boundary["max_flops_per_sample"] is None:
        out = R / "results" / "000"
        out.mkdir(parents=True, exist_ok=True)
        size, err = sandbox_size(R, "000", out)
        if size is None:
            raise SystemExit(f"base model size check failed (fix base/ or data_loader.example_inputs):\n{err}")
        f = float(budget["size_factor"])
        boundary["max_params"] = boundary["max_params"] or int(f * size["params"])
        boundary["max_flops_per_sample"] = boundary["max_flops_per_sample"] or int(f * size["flops"])
        jsave(R / "boundary.json", boundary)
    res = train_candidate(R, "000")
    if res.get("failed"):
        raise SystemExit(f"the base model (candidate 000) failed, so the search cannot start:\n{res.get('error', '')[-2000:]}")
    if boundary["timeout_s"] is None:   # improve mode: timeouts from the base model's measured train time
        t0 = float(res.get("train_time_s") or res.get("wall_s") or 600)
        boundary["timeout_s"] = int(budget["timeout_ratio_to_H_time"] * t0) + 300
        if "final" in boundary and boundary["final"]["timeout_s"] is None:
            scale = boundary["final"]["train_samples"] / max(1, budget["train_samples"])
            boundary["final"]["timeout_s"] = int(budget["timeout_ratio_to_H_time"] * t0 * scale) + 300
        jsave(R / "boundary.json", boundary)
    if budget["sigma_seed"] == "auto":
        # σ = s.d. of the base score over seeds 0,1,2 (two extra base trainings; floor 1e-4)
        vals = [res["score"]]
        for sd in (1, 2):
            r2 = run_job.run(C, R / "cands" / "000", R / "results" / f"000_seed{sd}", budget["train_samples"], "val",
                             sd, "{}", timeout_s=boundary["timeout_s"], trainer=boundary["trainer_path"],
                             loader=boundary["loader_path"])
            if not r2.get("failed") and r2.get("score") is not None and math.isfinite(r2["score"]):
                vals.append(internal(boundary, r2["score"]))
        m = sum(vals) / len(vals)
        sig = max(1e-4, (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5) if len(vals) > 1 else 0.01
        st = jload(R / "state.json")
        st["sigma"], st["sigma_from"] = sig, {"seeds": len(vals), "scores": vals}
        st["reward_margin"] = float(budget.get("reward_margin_sigma", 0)) * sig
        jsave(R / "state.json", st)
        event(R, "sigma_measured", sigma=sig, scores=vals)
    regen_ledger(R)
    return {"run_dir": str(R), "base": res, "sigma": jload(R / "state.json")["sigma"]}


# ---------------------------------------------------------------- candidate checks + training
def static_check(R, cid):
    d = R / "cands" / cid
    extra = [p.name for p in d.iterdir() if p.name not in CAND_ALLOWED]
    missing = [f for f in EDITABLE + ["notes.json"] if not (d / f).exists()]
    if extra or missing:
        return f"candidate dir must hold only {sorted(CAND_ALLOWED - {'__pycache__', 'result.json'})}; extra={extra} missing={missing}"
    try:
        json.load(open(d / "notes.json"))
    except Exception as e:  # noqa: BLE001
        return f"notes.json is not valid JSON: {e}"
    h = code_hash(d)
    for other in cand_ids(R):
        if other != cid and (R / "cands" / other / "result.json").exists() and code_hash(R / "cands" / other) == h:
            return f"duplicate: model.py and train_config.py are identical to candidate {other}; make a real change"
    return None


def code_hash(d):
    import hashlib
    return hashlib.sha256(b"".join(re.sub(rb"#[^\n]*|\s+", b"", (d / f).read_bytes()) for f in EDITABLE)).hexdigest()


def sandbox_size(R, cid, out):
    case_dir = PROJECT / "cases" / jload(R / "state.json")["case"]
    B = jload(R / "boundary.json", {}) or {}
    ro = {str(case_dir): "/case"}
    if B.get("loader_path"):
        ro[B["loader_path"]] = "/case/data_loader.py"
    ro.update({str(R / "cands" / cid): "/cand", str(HERE / "check_size.py"): "/harness/check_size.py",
               **run_job.extra_ro(case_dir)})
    mounts = sandbox.Mounts(ro=ro, rw={str(out): "/out"})
    argv = sandbox.build_argv([run_job.sys.executable, "-I", "/harness/check_size.py"], mounts=mounts,
                              cwd="/out", threads=4, scratch=out / ".scratch")
    with open(out / "size.log", "w") as log:
        p = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    shutil.rmtree(out / ".scratch", ignore_errors=True)
    if p.returncode != 0 or not (out / "size.json").exists():
        return None, (out / "size.log").read_text()[-3000:]
    return jload(out / "size.json"), None


def train_candidate(R, cid):
    """Check boundary and size, train in the sandbox at the fixed budget, write cands/<cid>/result.json."""
    B = jload(R / "boundary.json")
    out = R / "results" / cid
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    res = None
    err = static_check(R, cid)
    if err is None:
        size, err = sandbox_size(R, cid, out)
        if size is not None:
            if size["params"] > B["max_params"] or size["flops"] > B["max_flops_per_sample"]:
                err = (f"size limit exceeded: params {size['params']} (max {B['max_params']}), "
                       f"flops/sample {size['flops']} (max {B['max_flops_per_sample']})")
    if err is None:
        try:
            res = run_job.run(PROJECT / "cases" / B["case"], R / "cands" / cid, out, B["train_samples"], "val", 0,
                              "{}", timeout_s=B["timeout_s"], trainer=B.get("trainer_path"), loader=B.get("loader_path"))
        except subprocess.TimeoutExpired:
            err = f"training exceeded the timeout ({B['timeout_s']} s)"
        if res and res.get("failed"):
            err = "training failed:\n" + Path(res["log"]).read_text()[-3000:]
        elif res and not math.isfinite(res.get("score", float("nan"))):
            err = "score is not finite (diverged or constant prediction)"
    if err is None:
        res = {k: v for k, v in res.items() if k != "per_cell_type"} | {"size": size}
        if not B.get("higher_is_better", True):   # lower-is-better metric: the loop maximises the negated score
            res["score_raw"], res["score"] = res["score"], -res["score"]
    else:
        res = {"failed": True, "error": err}
    jsave(R / "cands" / cid / "result.json", res)
    event(R, "trained", cand=cid, score=res.get("score"), error=res.get("error", "")[:300])
    return res


# ---------------------------------------------------------------- space
def validate_space(R, k):
    sp = jload(R / f"space_v{k}.json")
    if sp is None:
        return f"space_v{k}.json was not written"
    axes = sp.get("axes")
    if not isinstance(axes, dict) or not 3 <= len(axes) <= 6:
        return "need 3–6 axes"
    for a, v in axes.items():
        ch = v.get("choices") if isinstance(v, dict) else None
        if not isinstance(ch, list) or not 2 <= len(ch) <= 6:
            return f"axis {a}: need 2–6 choices"
    base = sp.get("base")
    if not isinstance(base, dict) or set(base) != set(axes) or any(base[a] not in axes[a]["choices"] for a in axes):
        return "'base' must map every axis to one of its choices (the base configuration must stay expressible)"
    if not isinstance(sp.get("off_axis"), dict):
        return "'off_axis' must be an object with 'allowed' and 'limits'"
    B = jload(R / "boundary.json")
    if B.get("ssa_protocol", 1) >= 2:   # granularity decision (user decision 2026-11-20)
        req = ("problem_type", "analogs", "templates", "granularity") if B.get("mode", "reproduce") == "reproduce" \
            else ("granularity",)   # improve mode (2026-11-22): analog/template fields are optional
        for key in req:
            if not sp.get(key):
                return f"ssa_protocol 2: '{key}' is required"
        fx = sp.get("fixed")
        if not isinstance(fx, list) or not all(isinstance(f, dict) and f.get("module") and f.get("reason") for f in fx):
            return "ssa_protocol 2: 'fixed' must list {module, choice, reason} (empty list only if every module is searched)"
        hp = sp.get("hparams")
        if not isinstance(hp, dict) or any(k not in hp for k in ("lr", "optimizer", "weight_decay", "scheduler")):
            return "ssa_protocol 2: 'hparams' must give ranges for lr, optimizer, weight_decay, scheduler (the exploiter's space)"
        extra = sum(len(v["choices"]) - 1 for v in axes.values())
        lim = module_budget(R)
        if extra > lim:
            return (f"ssa_protocol 2: {extra} non-base choices > remaining candidates/2 = {lim}; "
                    "fix more modules or drop choices")
    cap = B.get("search_date_cap")
    for a, v in (axes.items() if cap else []):
        for s in v.get("sources", []):
            m = re.search(r"\((\d{4}-\d{2}-\d{2})\)\s*$", str(s))
            if not m:
                return f"axis {a}: source '{s}' needs a verified date 'url (YYYY-MM-DD)'"
            if m.group(1) >= cap:
                return f"axis {a}: source '{s}' is dated on/after the search date cap {cap}"
    return None


def module_budget(R):
    """Max Σ(choices − 1) for a new space: half the candidates still left, at least 2 (review fix 2026-11-20)."""
    B = jload(R / "boundary.json")
    left = B["candidates_per_run"] - (len(cand_ids(R)) - 1)
    return max(3, left // 2)   # ≥ 3: the axis rules (≥ 3 axes, ≥ 2 choices) need at least 3 non-base choices (fix 2026-11-22)


def coverage(R, space, since=None):
    """Fraction of (axis, choice) pairs used by ≥1 trained candidate made under this space (id ≥ since).
    Uses the candidates' effective module choices (inherited from the parent chain, base = space['base'])."""
    if since is None:
        since = jload(R / "state.json").get("space_first_id", 0)
    counts = {a: {c: 0 for c in v["choices"]} for a, v in space["axes"].items()}
    ea = eff_axes(R, space)
    for cid, _, _ in scored(R):
        if cid.isdigit() and int(cid) < since:
            continue
        for a, c in ea.get(cid, {}).items():
            if a in counts and c in counts[a]:
                counts[a][c] += 1
    # protocol 2: every candidate inherits the base's choices, so only non-base choices count toward coverage
    bse = (space.get("base") or {}) if (jload(R / "boundary.json", {}) or {}).get("ssa_protocol", 1) >= 2 else {}
    cells = [n for a in counts for c, n in counts[a].items() if bse.get(a) != c]
    return (sum(n > 0 for n in cells) / len(cells) if cells else 0.0), counts


def choice_gains(R, sigma):
    """{(axis, choice): [gain in σ]} from explorer candidates that targeted that coordinate.
    Gain = best score of the explorer candidate or its exploiter-only descendants (the module after tuning, review
    decision C 2026-11-20) minus the parent's score, in σ, clipped to ±3. A failed candidate with no descendants = −3."""
    sc = scored(R)
    score = {cid: sv for cid, sv, _ in sc}
    out = {}
    p = R / "events.jsonl"
    for line in (p.read_text().splitlines() if p.exists() else []):
        e = json.loads(line)
        if e.get("kind") != "candidate":
            continue
        d = e.get("decision") or {}
        t = d.get("target")
        if d.get("agent") != "explorer" or not isinstance(t, dict) or len(t) != 1:
            continue
        (a, c), = t.items()
        ps = score.get(d.get("parent"))
        x = e.get("cand")
        tuned = [sv for cid, sv, _ in sc if x and lineage_ok(R, cid, x)]
        s_ = max(tuned) if tuned else None
        g = -REWARD_CLIP if s_ is None or ps is None else max(-REWARD_CLIP, min(REWARD_CLIP, (s_ - ps) / sigma))
        out.setdefault((a, c), []).append(g)
    return out


# ---------------------------------------------------------------- allocator
def allocate(R):
    st = jload(R / "state.json")
    B = jload(R / "boundary.json")
    sc = scored(R)
    n_cands = len(cand_ids(R)) - 1
    k, space = latest_space(R, st)
    if n_cands >= B["candidates_per_run"]:
        return {"action": "stop", "why": "candidate budget reached"}
    closed = [r for r in st["ssa_log"] if "improved" in r]          # the open (current) round has no verdict yet
    if len(closed) >= STOP_SSA_ROUNDS + 1 and all(not r["improved"] for r in closed[-STOP_SSA_ROUNDS:]):
        return {"action": "stop", "why": f"{STOP_SSA_ROUNDS} SSA rounds without improvement"}
    if space is None:
        return {"action": "ssa", "why": "start"}
    cov, counts = coverage(R, space)
    # outer loop cap (user decision 2026-11-20): at most MAX_SSA_REVISIONS revisions after the first space;
    # after that the run continues on the last space until the candidate budget
    # stagnation: last K candidates did not raise the best by DELTA·σ
    order = [s for s in sc if s[0] != "000"]
    # fix 2026-11-22: no revision once one has failed, or with fewer than MIN_LEFT_FOR_SSA candidates left (a new space
    # needs ≥ 2 candidates per open choice and the rules keep ≥ 3 choices open, so a late revision cannot be valid)
    left = B["candidates_per_run"] - n_cands
    can_revise = (len(st["ssa_log"]) - 1 < MAX_SSA_REVISIONS and not st.get("ssa_frozen")
                  and left >= MIN_LEFT_FOR_SSA)
    if can_revise and len(order) >= K_STAGNATION and st["since_ssa"] >= K_STAGNATION:
        before = max([s[1] for s in sc if s not in order[-K_STAGNATION:]] or [-1])
        if max(s[1] for s in order[-K_STAGNATION:]) < before + DELTA * st["sigma"]:
            return {"action": "ssa", "why": f"stagnation ({K_STAGNATION} candidates < best + {DELTA}σ)"}
    if can_revise and cov >= COVERAGE_C and st["since_ssa"] > 0:
        return {"action": "ssa", "why": f"coverage {cov:.2f} ≥ {COVERAGE_C}"}
    if can_revise and st["since_ssa"] >= MAX_INTERVAL_M:
        return {"action": "ssa", "why": f"max interval {MAX_INTERVAL_M}"}
    # bandit (UCB1) over exploiter / explorer
    bd = st["bandit"]
    total = sum(n for n, _ in bd.values())
    hist = st.get("bandit_hist") if st.get("reward_mode", "step") == "delta" else None
    def ucb(arm):
        n, s = bd[arm]
        if n == 0:
            return float("inf")
        # delta mode: mean of the agent's last REWARD_WINDOW rewards (sliding window), bonus from all turns
        mean = (sum(hist[arm][-REWARD_WINDOW:]) / len(hist[arm][-REWARD_WINDOW:])) if hist and hist.get(arm) else s / n
        return mean + UCB_C * math.sqrt(math.log(max(total, 1)) / n)
    # allocation policy (bandit validation, 2026-11-19): AGH_POLICY at init-run -> state["policy"]
    pol = st.get("policy", "ucb")
    if pol == "alternate":
        arm = "exploiter" if total % 2 == 0 else "explorer"
    elif pol in ("exploiter", "explorer"):
        arm = pol
    else:
        arm = max(["exploiter", "explorer"], key=lambda a: (ucb(a), random.random()))
        # quota (fix 2026-11-22, V3 fid_ucb_r1: one −3 reward kept the explorer out for 29 rounds): an agent that made
        # fewer than ARM_MIN of the last ARM_WINDOW candidates is called next. Only for runs started with the fix.
        if "reward_clip" in st:
            recent = [n.get("agent") for n in (jload(R / "cands" / c / "notes.json", {}) or {} for c in cand_ids(R)[1:])][-ARM_WINDOW:]
            if len(recent) >= ARM_WINDOW // 2:
                low = [a for a in ("exploiter", "explorer") if recent.count(a) < ARM_MIN]
                if low:
                    arm = low[0]
    ranked = sorted(sc, key=lambda s: -s[1])
    p2 = B.get("ssa_protocol", 1) >= 2
    if arm == "exploiter":
        kids = {}
        for cid in cand_ids(R):
            n = jload(R / "cands" / cid / "notes.json", {})
            if n.get("agent") == "exploiter":
                kids[n.get("parent")] = kids.get(n.get("parent"), 0) + 1
        pool = ranked[:3]
        if p2:
            # review decision C (2026-11-20): a new module is judged after tuning, so an untuned explorer candidate
            # within 2σ of the best also qualifies for one exploiter turn
            best = ranked[0][1]
            pool += [s for s in ranked[3:] if s[2].get("agent") == "explorer" and kids.get(s[0], 0) == 0
                     and s[1] >= best - 2 * st["sigma"]]
        parent = min(pool, key=lambda s: (kids.get(s[0], 0), -s[1]))[0]
        return {"action": "candidate", "agent": "exploiter", "parent": parent, "space": k}
    # explorer: least-covered (axis, choice); parent = best candidate not already at that coordinate
    if random.random() < P_OFF_AXIS:
        return {"action": "candidate", "agent": "explorer", "parent": ranked[0][0], "target": "off-axis", "space": k}
    ea = eff_axes(R, space) if p2 else {s[0]: (s[2].get("axes") or {}) for s in sc}
    if p2:
        # protocol 2 (user decision 2026-11-20): UCB over (axis, choice). gain of an explorer candidate =
        # clip((best tuned S − S_parent)/σ, ±3), failure = −3; untried choices first; bonus √(2 ln N / n).
        # Only choices some scored model lacks can be targets (review fix: the base's own modules are not "untried").
        stats = choice_gains(R, st["sigma"])
        N = max(1, sum(len(v) for v in stats.values()))
        present = {(a, c) for s in sc for a, c in ea.get(s[0], {}).items()}
        def cucb(ac):
            g = stats.get(ac, [])
            if not g:   # never targeted: untried modules first; a choice some model already has counts as gain 0
                return float("inf") if ac not in present else UCB_C * math.sqrt(math.log(N + 1))
            return sum(g) / len(g) + UCB_C * math.sqrt(math.log(N + 1) / len(g))
        opts = [(a, c) for a in counts for c in counts[a] if any(ea.get(s[0], {}).get(a) != c for s in sc)]
        if not opts:
            return {"action": "candidate", "agent": "explorer", "parent": ranked[0][0], "target": "off-axis", "space": k}
        a, c = max(opts, key=lambda ac: (cucb(ac), -counts[ac[0]][ac[1]], random.random()))
    else:
        a, c = min(((a, c) for a in counts for c in counts[a]), key=lambda ac: (counts[ac[0]][ac[1]], random.random()))
    parent = next((s[0] for s in ranked if ea.get(s[0], {}).get(a) != c), ranked[0][0])
    return {"action": "candidate", "agent": "explorer", "parent": parent, "target": {a: c}, "space": k}


# ---------------------------------------------------------------- agents
def _err1(r):
    return (r.get("error") or "").strip().splitlines()[-1][:160].replace("|", "/") if r.get("error") else ""


def write_digest(R, role, dec=None):
    """Spec of what each agent remembers: .claude/agent-memory/<role>/MEMORY.md (keep the two in sync).
    Per-call memory (user request 2026-11-20): the loop summarises the run history that matters for THIS call into
    runs/<case>/<run>/memory/<role>.md, so the agent starts from a focused summary instead of re-deriving it from
    the full LEDGER. Built by code from tree + events only (no answer keys, no other runs)."""
    st = jload(R / "state.json")
    B = jload(R / "boundary.json", {}) or {}
    k, space = latest_space(R, st)
    sig = st["sigma"]
    sc = scored(R)
    ranked = sorted(sc, key=lambda s: -s[1])
    ea = eff_axes(R, space) if space else {}
    res = {cid: jload(R / "cands" / cid / "result.json", {}) or {} for cid in cand_ids(R)}
    notes = {cid: jload(R / "cands" / cid / "notes.json", {}) or {} for cid in cand_ids(R)}
    L = [f"# Memory for this {role} call (written by the loop; summary of this run only)", ""]
    if ranked:
        b = ranked[0]
        L.append(f"Best: {b[0]} S_val {b[1]:.4f}. σ (seed noise) = {sig:.4f}. "
                 f"Candidates left: {B.get('candidates_per_run', 0) - (len(cand_ids(R)) - 1)}.")
        tgt = B.get("target")
        if tgt:
            ti = internal(B, tgt["value"])   # lower-is-better targets are compared on the negated scale
            L.append(f"Target {tgt['value']}: gap {ti - b[1]:+.4f} ({(ti - b[1]) / sig:+.1f} σ).")
    L.append("")

    def mods(cid):
        return ", ".join(f"{a}={c}" for a, c in ea.get(cid, {}).items()) or "(base)"

    if role == "exploiter" and dec:
        p = dec["parent"]
        same = [c for c in cand_ids(R) if ea.get(c) == ea.get(p) and res[c]]
        hp = {c: hp_of(R / "cands" / c) for c in same}
        L += [f"## Architecture of parent {p}: {mods(p)}", "",
              f"Hyperparameter history for this architecture ({len(same)} trained):", "",
              "| id | hyperparameters (≠ parent) | S_val | Δ vs parent (σ) | status |", "|---|---|---|---|---|"]
        ps = res.get(p, {}).get("score")
        for c in same:
            d = ", ".join(f"{a}={v}" for a, v in hp[c].items() if hp[p].get(a) != v) or "= parent"
            sv = res[c].get("score")
            dz = f"{(sv - ps) / sig:+.1f}" if (sv is not None and ps is not None) else ""
            L.append(f"| {c} | {d} | {'' if sv is None else f'{sv:.4f}'} | {dz} | {'FAILED: ' + _err1(res[c]) if res[c].get('failed') else 'ok'} |")
        # evidence pairs: same architecture, exactly one hyperparameter differs, |ΔS| ≥ 2σ (review decision A)
        ev = []
        for i, a in enumerate(same):
            for b2 in same[i + 1:]:
                diff = [x for x in set(hp[a]) | set(hp[b2]) if hp[a].get(x) != hp[b2].get(x)]
                sa, sb = res[a].get("score"), res[b2].get("score")
                if len(diff) == 1 and sa is not None and sb is not None and abs(sa - sb) >= 2 * sig:
                    w, l_ = (a, b2) if sa > sb else (b2, a)
                    ev.append(f"- {diff[0]}: {hp[w].get(diff[0])} ({w}) beats {hp[l_].get(diff[0])} ({l_}) by {abs(sa - sb) / sig:.1f} σ")
        L += ["", "Evidence for narrowing (one hyperparameter differs, gap ≥ 2σ):"] + (ev or ["- none yet: keep full ranges"])
        tried = {x.lower() for c in same for x in hp[c] if hp[c].get(x) != hp[p].get(x)}
        untried = [h for h in ((space or {}).get("hparams") or {}) if h.lower() not in tried]
        L += ["", "Hyperparameters in the space not yet changed for this architecture: " + (", ".join(untried) or "none")]
    elif role == "explorer" and dec:
        t = dec.get("target")
        stats = choice_gains(R, sig)
        L += [f"## Your target: {json.dumps(t)}; parent {dec['parent']} ({mods(dec['parent'])})", "",
              "Module results so far (gain = best score after tuning − parent, in σ, clipped ±3; −3 = failed):", "",
              "| axis | choice | tries | mean gain (σ) |", "|---|---|---|---|"]
        for (a, c), g in sorted(stats.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            L.append(f"| {a} | {c} | {len(g)} | {sum(g) / len(g):+.1f} |")
        fails = [c for c in cand_ids(R) if res[c].get("failed") and notes[c].get("agent") == "explorer"]
        L += ["", "Failed explorer candidates (learn from these):"] + \
             ([f"- {c} ({notes[c].get('replaced') or notes[c].get('target') or ''}): {_err1(res[c])}" for c in fails] or ["- none"])
        if isinstance(t, dict):
            (a, c), = t.items()
            prior = [x for x in cand_ids(R) if ea.get(x, {}).get(a) == c and x != "000"]
            L += ["", f"Earlier candidates with {a}={c}: " + (", ".join(
                f"{x} ({'FAILED' if res[x].get('failed') else res[x].get('score') and round(res[x]['score'], 4)})" for x in prior) or "none")]
        offs = [c for c in cand_ids(R) if str(notes[c].get("target")) == "off-axis"]
        L += ["", "Off-axis modules already tried: " + (", ".join(f"{c}: {str(notes[c].get('off_axis'))[:60]}" for c in offs) or "none")]
    elif role == "search-space":
        L += ["## Space revisions so far", ""]
        for r in st["ssa_log"]:
            L.append(f"- v{r['k']} ({r['why']}): best at start {r.get('best_at_start')}, improved: {r.get('improved', 'open')}")
        if space:
            L += ["", "## Per module choice (all candidates in this run)", "",
                  "| axis | choice | trained | failed | best S_val | mean tuned gain (σ) |", "|---|---|---|---|---|---|"]
            stats = choice_gains(R, sig)
            for a, v in space["axes"].items():
                for c in v["choices"]:
                    xs = [x for x in cand_ids(R) if ea.get(x, {}).get(a) == c and res[x]]
                    ok = [res[x]["score"] for x in xs if res[x].get("score") is not None]
                    g = stats.get((a, c))
                    L.append(f"| {a} | {c} | {len(ok)} | {len(xs) - len(ok)} | {max(ok):.4f} | " if ok else f"| {a} | {c} | 0 | {len(xs)} | | ")
                    L[-1] += (f"{sum(g) / len(g):+.1f} |" if g else " |")
        L += ["", "## Top 5 models", ""] + [f"- {c} {sv:.4f}: {mods(c)}" for c, sv, _ in ranked[:5]]
        gaps = [(c, notes[c].get("space_gap")) for c in cand_ids(R)]
        gaps = [f"- {c}: {str(g)[:200]}" for c, g in gaps if g and str(g).lower() not in ("none", "no", "")]
        L += ["", "## Explorer notes: modules missing from the space", ""] + (gaps or ["- none"])
    steer = R / "steer.md"   # user steering (2026-11-22): edit runs/<case>/<run>/steer.md while the run goes
    if steer.exists() and steer.read_text().strip():
        L = L[:2] + ["## Instructions from the user for this run (follow them)", "", steer.read_text().strip()[:4000], ""] + L[2:]
    d = R / "memory"
    d.mkdir(exist_ok=True)
    (d / f"{role}.md").write_text("\n".join(L) + "\n")
    return d / f"{role}.md"


LAST_CALL = None


def finish_call(path, outcome):
    """Attach what the call produced (score, error, space validity) to its log record, then upload it to Weave
    in the background (unless AGH_WEAVE=0). Never blocks or breaks the loop."""
    if path is None or not Path(path).exists():
        return
    rec = jload(path)
    rec["outcome"] = outcome
    jsave(path, rec)
    if os.environ.get("AGH_WEAVE", "1") == "0":
        return
    wpy = os.environ.get("AGH_WEAVE_PY") or str(Path.home() / "miniforge3/bin/python")
    if not Path(wpy).exists():   # no Python with `weave` on this server: keep the local record only
        return
    try:
        subprocess.Popen([wpy, str(HERE / "weave_log.py"), str(path)], stdout=subprocess.DEVNULL,
                         stderr=open(Path(path).parent / "weave_upload.log", "a"), start_new_session=True)
    except OSError:
        pass


def call_agent(R, role, prompt, meta=None):
    """Run one agent call. Full input/output is logged (user request 2026-11-20):
    runs/<case>/<run>/calls/<seq>_<role>.json = prompt, call metadata (candidate, decision), the memory file the agent was
    pointed to, the whole stream-json transcript (every message and tool call/result), final reply, cost and usage.
    If AGH_WEAVE is not "0", harness/weave_log.py uploads each record to W&B Weave in the background."""
    env = dict(os.environ, AGH_ROLE=role, AGH_RUN_DIR=str(R), WEAVE_AGENT_NAME=f"agh-{role}")
    tools = {"search-space": "Read,Glob,Grep,Write,WebSearch,WebFetch"}.get(role, "Read,Glob,Grep,Write,Edit")
    cmd = ["claude", "-p", "--agent", role, "--model", os.environ.get("AGH_MODEL", "opus"),
           "--permission-mode", "acceptEdits", "--allowedTools", tools, "--output-format", "stream-json", "--verbose",
           prompt]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=PROJECT, env=env, capture_output=True, text=True, timeout=3600)
    transcript, out = [], None
    for line in p.stdout.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "system" and e.get("subtype", "").startswith("hook"):
            continue
        transcript.append(e)
        if e.get("type") == "result":
            out = e
    if out is None:
        out = {"result": p.stdout[-2000:], "is_error": True, "stderr": p.stderr[-2000:]}
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "secs": round(time.time() - t0, 1),
           "cost_usd": out.get("total_cost_usd"), "usage": out.get("usage"), "is_error": out.get("is_error"),
           "num_turns": out.get("num_turns")}
    with open(R / "tokens.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    cdir = R / "calls"
    cdir.mkdir(exist_ok=True)
    seq = len(list(cdir.glob("*.json"))) + 1
    m = re.search(r"Read (\S+/memory/[\w-]+\.md) first", prompt)
    mem = (PROJECT / m.group(1)).read_text() if m and (PROJECT / m.group(1)).exists() else None
    st = jload(R / "state.json", {}) or {}
    call = {"seq": seq, **rec, "case": st.get("case"), "run": st.get("run"), "meta": meta or {}, "prompt": prompt,
            "memory": mem, "reply": out.get("result", ""), "session_id": out.get("session_id"),
            "stderr": p.stderr[-2000:] if p.returncode else "", "transcript": transcript}
    path = cdir / f"{seq:04d}_{role}.json"
    jsave(path, call)
    global LAST_CALL
    LAST_CALL = path
    return out.get("result", "")


def rel(R):
    return str(R.relative_to(PROJECT))


def run_ssa(R, why):
    st = jload(R / "state.json")
    k = (st["spaces"][-1] + 1) if st["spaces"] else 1
    best_before = max((s[1] for s in scored(R)), default=None)
    B = jload(R / "boundary.json", {}) or {}
    msg = (f"Run directory: {rel(R)}. Write the search space version k={k} to {rel(R)}/space_v{k}.json. "
           f"Reason for this call: {why}. Include a 'base' object mapping every axis to the base model's choice.")
    if B.get("ssa_protocol", 1) >= 2:
        left = B["candidates_per_run"] - (len(cand_ids(R)) - 1)
        msg += (f" Candidates left in this run: {left}; module budget Σ(choices − 1) ≤ {module_budget(R)}.")
    tgt = B.get("target")
    if tgt and best_before is not None and k > 1:
        # target gap (review decision B, 2026-11-20): information for the SSA; WIDEN only when far from the target
        # AND the last revision brought no gain (the target is a test number from another setup, so a gap alone
        # does not prove the space is too narrow)
        gap = (internal(B, tgt["value"]) - best_before) / st["sigma"]
        last = st["ssa_log"][-1] if st["ssa_log"] else {}
        last_gain = (best_before > last["best_at_start"] + DELTA * st["sigma"]) if last.get("best_at_start") is not None else True
        if gap > TARGET_WIDEN_SIGMA and not last_gain:
            direction = ("WIDEN: far below the target and the current space brought no gain; open a fixed module "
                         "that looks like the bottleneck or add module choices from similar fields")
        elif gap > TARGET_WIDEN_SIGMA:
            direction = ("KEEP: far below the target but the current space is still improving; keep its open modules, "
                         "fix only axes the LEDGER has clearly decided")
        else:
            direction = "NARROW: close to or above the target; fix decided axes and tighten hparams around the best values"
        what = ("the target is a published test number" if B.get("mode", "reproduce") == "reproduce"
                else "the target is the user's goal")
        msg += (f" Target gap: best S_val {best_before:.4f} vs target {internal(B, tgt['value'])} = {gap:+.1f} σ "
                f"({what}; treat the gap as information). Direction: {direction}.")
    err = None
    for attempt in range(MAX_RETRIES + 1):
        mp = write_digest(R, "search-space")
        reply = call_agent(R, "search-space", (msg if err is None else msg + f"\nYour previous file was invalid: {err}. Fix it.")
                           + f" Read {rel(mp)} first: the loop's summary of this run for you.",
                           meta={"space_version": k, "why": why, "attempt": attempt})
        err = validate_space(R, k)
        finish_call(LAST_CALL, {"space_version": k, "valid": err is None, "error": err})
        if err is None:
            break
    event(R, "ssa", k=k, why=why, error=err, reply=reply[-1500:])
    if err is not None:
        if not st["spaces"]:
            raise SystemExit(f"SSA failed to produce a valid space after retries: {err}")
        # fix 2026-11-22: a failed revision no longer kills the run (V2 fid_ucb_r1 and V6 fid_ucb_r2 died here with
        # 25/28 candidates and no finalize). Keep the last valid space, stop revising, continue to the budget.
        st["ssa_frozen"] = {"k": k, "why": why, "error": err}
        st["since_ssa"] = 0
        jsave(R / "state.json", st)
        event(R, "ssa_failed_keep_last_space", k=k, kept=st["spaces"][-1])
        return
    if st["ssa_log"]:
        st["ssa_log"][-1]["improved"] = (max((s[1] for s in scored(R)), default=-1)
                                         > st["ssa_log"][-1]["best_at_start"] + DELTA * st["sigma"])
    st["ssa_log"].append({"k": k, "why": why, "best_at_start": best_before})
    st["spaces"].append(k)
    # review fix 2026-11-20: record the base model's module choices in 000's notes so agents reading notes see them
    n0 = jload(R / "cands" / "000" / "notes.json", {}) or {}
    n0["axes"] = dict(jload(R / f"space_v{k}.json").get("base") or {})
    jsave(R / "cands" / "000" / "notes.json", n0)
    st["space_first_id"] = st["next_id"]       # coverage counts only candidates made under the new space
    st["since_ssa"] = 0
    jsave(R / "state.json", st)
    regen_ledger(R)


def run_candidate_step(R, dec):
    st = jload(R / "state.json")
    cid = f"{st['next_id']:03d}"
    st["next_id"] += 1
    jsave(R / "state.json", st)
    target = dec.get("target")
    tgt = "" if target is None else f" Target coordinate: {json.dumps(target)}."
    msg = (f"Run directory: {rel(R)}. Candidate id: {cid}. Parent: {dec['parent']}. "
           f"Current space: {rel(R)}/space_v{dec['space']}.json.{tgt} Write the candidate into {rel(R)}/cands/{cid}/.")
    best_before = max((s[1] for s in scored(R)), default=-1)
    if jload(R / "boundary.json").get("ssa_protocol", 1) >= 2:
        msg += f" Read {rel(write_digest(R, dec['agent'], dec))} first: the loop's summary of this run for your call."
    res, reply = None, ""
    for attempt in range(MAX_RETRIES + 1):
        if attempt == 0:
            reply = call_agent(R, dec["agent"], msg, meta={"cand": cid, "decision": dec, "attempt": 0})
        else:
            reply = call_agent(R, dec["agent"], meta={"cand": cid, "decision": dec, "attempt": attempt},
                               prompt=msg + f"\nThe candidate failed (attempt {attempt}). Error:\n"
                               f"{res['error']}\nFix the files in {rel(R)}/cands/{cid}/ (same id).")
        (R / "cands" / cid).mkdir(parents=True, exist_ok=True)
        res = train_candidate(R, cid)
        finish_call(LAST_CALL, {"cand": cid, "S_val": res.get("score"), "failed": bool(res.get("failed")),
                                "error": (res.get("error") or "")[:2000], "best_before": best_before})
        if not res.get("failed"):
            break
    parent_score = (jload(R / "cands" / dec["parent"] / "result.json", {}) or {}).get("score", -1)
    s = res.get("score")
    st = jload(R / "state.json")
    m = st.get("reward_margin", 0.0)
    if st.get("reward_mode", "step") == "delta":
        # reward = score − best score before this candidate (user decision 2026-11-20); a failed candidate counts
        # as if it scored the base model A0
        # reward = clip((score − best) / σ, −3, +3) (user decision 2026-11-20): σ makes the reward comparable with the
        # UCB bonus; the clip stops one broken candidate from sinking an agent's mean
        a0 = (jload(R / "cands" / "000" / "result.json", {}) or {}).get("score", best_before)
        lo, hi = st.get("reward_clip", [-REWARD_CLIP, REWARD_CLIP])
        if s is None and "reward_fail" in st:
            reward = st["reward_fail"]
        else:
            reward = max(lo, min(hi, ((s if s is not None else a0) - best_before) / st["sigma"]))
    else:
        reward = 0.0 if s is None else (1.0 if s > best_before + m else 0.5 if s > parent_score + m else 0.0)
    n, tot = st["bandit"][dec["agent"]]
    st["bandit"][dec["agent"]] = [n + 1, tot + reward]
    st.setdefault("bandit_hist", {"exploiter": [], "explorer": []})[dec["agent"]].append(reward)
    st["since_ssa"] += 1
    jsave(R / "state.json", st)
    if jload(R / "boundary.json").get("ssa_protocol", 1) >= 2 and dec["agent"] == "exploiter":
        _, sp = latest_space(R, st)
        ea = eff_axes(R, sp)
        if ea.get(cid) != ea.get(dec["parent"]):   # exploiter must not change modules (logged, not rejected)
            event(R, "role_violation", cand=cid, agent="exploiter", parent_axes=ea.get(dec["parent"]), cand_axes=ea.get(cid))
    event(R, "candidate", cand=cid, decision=dec, score=s, reward=reward, reply=reply[-1500:])
    regen_ledger(R)


def loop(case, run):
    R = rdir(case, run)
    while True:
        dec = allocate(R)
        event(R, "decision", **dec)
        if dec["action"] == "stop":
            break
        if dec["action"] == "ssa":
            run_ssa(R, dec["why"])
        else:
            run_candidate_step(R, dec)
    fin = finalize(case, run)
    return {"stopped": dec["why"], "finalize": fin}


# ---------------------------------------------------------------- finalize (test once)
def finalize(case, run):
    R = rdir(case, run)
    st = jload(R / "state.json")
    if st.get("finalized"):
        return {"error": "already finalized; test is graded once", "final": jload(R / "final.json")}
    st["finalized"] = True              # lock before grading
    jsave(R / "state.json", st)
    B = jload(R / "boundary.json")
    ranked = sorted(scored(R), key=lambda s: -s[1])
    best, reselect = ranked[0], None
    # top-3 reseed (review decision D) is off by default since 2026-11-20: one training run per model (user decision,
    # training cost); enable per case with budget.yaml reselect_top3: true
    if B.get("reselect_top3") and len(ranked) > 1:
        # review decision D (2026-11-20): the val maximum is biased upward (winner's curse). Retrain the top 3 with
        # seed 1 on val and pick the final model by the mean of the two val scores. Test is still run once.
        reselect = []
        for cid, sv, _ in ranked[:3]:
            try:
                r1 = run_job.run(PROJECT / "cases" / case, R / "cands" / cid, R / "results" / f"reseed_{cid}",
                                 B["train_samples"], "val", 1, "{}", timeout_s=B["timeout_s"],
                                 trainer=B.get("trainer_path"), loader=B.get("loader_path"))
            except subprocess.TimeoutExpired:
                r1 = {"failed": True}
            s1 = internal(B, r1.get("score")) if not r1.get("failed") else None
            ok = s1 is not None and math.isfinite(s1)
            reselect.append({"cand": cid, "S_val_seed0": sv, "S_val_seed1": s1 if ok else None,
                             "mean": (sv + s1) / 2 if ok else sv - 1e9})
        pick = max(reselect, key=lambda r: r["mean"])
        best = next(s for s in ranked if s[0] == pick["cand"])
        event(R, "reselect", top3=reselect, picked=best[0])
    out = R / "results" / f"final_{best[0]}"
    fin = B.get("final") or {}
    try:
        res = run_job.run(PROJECT / "cases" / case, R / "cands" / best[0], out,
                          fin.get("train_samples", B["train_samples"]), "test", 0, "{}",
                          timeout_s=fin.get("timeout_s", B["timeout_s"]), extra_args=fin.get("extra_args", []),
                          trainer=B.get("trainer_path"), loader=B.get("loader_path"))
    except subprocess.TimeoutExpired:
        res = {"failed": True, "error": "final training exceeded the timeout"}
    final = {"best": best[0], "S_val": internal(B, best[1]), "S_test": res.get("score"), "failed": res.get("failed", False),
             "n_candidates": len(cand_ids(R)) - 1, "reselect": reselect, "final_training": fin or "search budget",
             "best_val_cand": ranked[0][0]}
    # report (user decision 2026-11-20: no separate judge stage; numbers for the report only, never used by the loop)
    from harness import judge as _rep
    tgt = B.get("target")
    if tgt and final["S_test"] is not None:
        final["target"] = tgt["value"]
        final["ratio_to_target"], final["verdict"] = _rep.reproduced(final["S_test"], tgt["value"],
                                                                     bool(tgt.get("higher_is_better", True)))
    final["leak"] = _rep.leak_check(R, case) if B.get("mode", "reproduce") == "reproduce" else {"leak": None}
    if B.get("trainer_path"):
        final["best_bundle"] = str(bundle_best(R, best[0], fin, B).relative_to(PROJECT))
    jsave(R / "final.json", final)
    event(R, "finalize", **final)
    return final


def bundle_best(R, cid, fin, B):
    """runs/<case>/<run>/best/ (user request 2026-11-22): everything to rerun the selected model's final training."""
    d = R / "best"
    d.mkdir(exist_ok=True)
    for f in EDITABLE + ["notes.json"]:
        if (R / "cands" / cid / f).exists():
            shutil.copy(R / "cands" / cid / f, d / f)
    shutil.copy(B["trainer_path"], d / Path(B["trainer_path"]).name)
    shutil.copy(B["loader_path"], d / "data_loader.py")
    for f in ("job.log", "result.json"):
        src = R / "results" / f"final_{cid}" / f
        if src.exists():
            shutil.copy(src, d / f"final_{f}")
    args = ["--train_samples", str(fin.get("train_samples", B["train_samples"])), "--eval", "test", "--seed", "0",
            "--model_kwargs", "'{}'", *map(str, fin.get("extra_args", []))]
    (d / "README.md").write_text(
        f"# Selected model {cid} of {B['case']}/{R.name}\n\n"
        f"Files: model.py, train_config.py (the candidate), {Path(B['trainer_path']).name} (the run's frozen trainer and scorer), "
        "data_loader.py (frozen), final_job.log / final_result.json (the one test evaluation).\n\n"
        "Rerun the final training (from the repo root, in the sandbox, same as the harness):\n\n"
        f"    harness/bin/agh rerun-best {B['case']} {R.name}\n\n"
        "The trainer's own command line (inside the sandbox /case = case folder, /cand = this folder, /out = output):\n\n"
        f"    python {Path(B['trainer_path']).name} {' '.join(args)}\n")
    return d


# ---------------------------------------------------------------- ledger
def regen_ledger(R):
    st = jload(R / "state.json")
    k, space = latest_space(R, st)
    B = jload(R / "boundary.json", {}) or {}
    lines = [f"# LEDGER — {st['case']} / {st['run']}", "",
             "Regenerated by the loop after every event. Do not edit.", "",
             "| id | agent | parent | axes | off-axis / idea | S_val | status |", "|---|---|---|---|---|---|---|"]
    p2 = B.get("ssa_protocol", 1) >= 2
    if p2:
        # review fixes 2026-11-20: modules = effective choices (inherited, base from the space); hparams are read
        # from the code (hp_of), shown as differences from the base model's code
        ea = eff_axes(R, space) if space else {}
        base_ax = ea.get("000", {})
        base_hp = hp_of(R / "cands" / "000")
        lines[-2:] = ["Modules and hyperparameters are read by the loop from each candidate's lineage and code; "
                      "only differences from the base model (000) are listed.", "",
                      "Base modules: " + (", ".join(f"{a}={c}" for a, c in base_ax.items()) or "(no space yet)"),
                      "Base hyperparameters: " + ", ".join(f"{a}={c}" for a, c in base_hp.items()).replace("|", "/"), "",
                      "| id | agent | parent | modules ≠ base | hparams ≠ base | new module / idea | S_val | status |",
                      "|---|---|---|---|---|---|---|---|"]
        groups = {}
    for cid in cand_ids(R):
        n = jload(R / "cands" / cid / "notes.json", {}) or {}
        r = jload(R / "cands" / cid / "result.json", {}) or {}
        status = ("FAILED: " + r.get("error", "")[:160].replace("\n", " ").replace("|", "/")) if r.get("failed") else ("ok" if r else "pending")
        axes = ", ".join(f"{a}={c}" for a, c in (n.get("axes") or {}).items())
        idea = (str(n.get("off_axis") or "") + " " + str(n.get("idea") or "")).strip().replace("|", "/")[:160]
        sv = f"{r['score']:.4f}" if r.get("score") is not None else ""
        if p2:
            mods = ", ".join(f"{a}={c}" for a, c in ea.get(cid, {}).items() if base_ax.get(a) != c) or "base"
            hp = hp_of(R / "cands" / cid)
            hps = (", ".join(f"{a}={c}" for a, c in hp.items() if base_hp.get(a) != c) or "base").replace("|", "/")
            idea2 = (str(n.get("replaced") or n.get("off_axis") or "") + " " + str(n.get("idea") or "")).strip().replace("|", "/")[:160]
            lines.append(f"| {cid} | {n.get('agent', '')} | {n.get('parent') or ''} | {mods} | {hps} | {idea2} | {sv} | {status} |")
            if r:
                groups.setdefault(mods, []).append((cid, hps, sv or "FAILED"))
        else:
            lines.append(f"| {cid} | {n.get('agent', '')} | {n.get('parent') or ''} | {axes} | {idea} | {sv} | {status} |")
    if p2 and groups:
        lines += ["", "## Hyperparameter history by architecture (same modules)", ""]
        for mods, rows in groups.items():
            lines += [f"**{mods}**: " + "; ".join(f"{c} [{h}] → {v}" for c, h, v in rows)]
    if p2:
        gaps = [(cid, (jload(R / "cands" / cid / "notes.json", {}) or {}).get("space_gap")) for cid in cand_ids(R)]
        gaps = [(c, str(g).replace("|", "/")[:200]) for c, g in gaps if g and str(g).lower() not in ("none", "no", "")]
        if gaps:
            lines += ["", "## Explorer notes: modules missing from the space", ""] + [f"- {c}: {g}" for c, g in gaps]
    if space:
        cov, counts = coverage(R, space)
        lines += ["", f"## Coverage of space_v{k}: {cov:.2f}", "", "| axis | choice | trained candidates |", "|---|---|---|"]
        for a in counts:
            for c, n in counts[a].items():
                lines.append(f"| {a} | {c} | {n} |")
    sc = scored(R)
    if sc:
        b = max(sc, key=lambda s: s[1])
        lines += ["", f"Best so far: {b[0]} (S_val {b[1]:.4f}). Seed noise σ ≈ {st['sigma']:.4f}."]
        tgt = B.get("target")
        if tgt:
            ti = internal(B, tgt["value"])
            lines += [f"Target: {tgt['value']} ({tgt['metric']}). Gap to target: {ti - b[1]:+.4f} "
                      f"({(ti - b[1]) / st['sigma']:+.1f} σ)."]
        if not B.get("higher_is_better", True):
            lines += ["Scores in this ledger are the NEGATED metric (lower-is-better metric; higher here = better)."]
    (R / "LEDGER.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- user helpers (2026-11-22)
def new_case(name):
    """Copy cases/_template to cases/<name>/ (generic trainer, toy data loader, budget with defaults)."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name or ""):
        raise SystemExit("case name: letters, digits, _ or - (e.g. my_model)")
    dst = PROJECT / "cases" / name
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    shutil.copytree(PROJECT / "cases" / "_template", dst, ignore=shutil.ignore_patterns("__pycache__", "data"))
    return {"created": str(dst.relative_to(PROJECT)),
            "next": [f"edit cases/{name}/data_loader.py, base/, task.md, budget.yaml (see cases/{name}/README.md)",
                     f"harness/bin/agh check-case {name}",
                     f"harness/bin/agh init-run {name} r1 && harness/bin/agh loop {name} r1"]}


def check_case(name, samples=None):
    """Before a run: budget check, sandboxed size check and a short sandboxed training of the base model
    (no agents). Writes runs/<name>/_check/ (overwritten each time)."""
    budget, mode = load_budget(name)
    C = PROJECT / "cases" / name
    R = rdir(name, "_check")
    shutil.rmtree(R, ignore_errors=True)
    (R / "cands" / "000").mkdir(parents=True)
    for f in EDITABLE:
        shutil.copy(C / "base" / f, R / "cands" / "000" / f)
    jsave(R / "state.json", {"case": name, "run": "_check"})
    jsave(R / "boundary.json", {"case": name})
    out = R / "results" / "000"
    out.mkdir(parents=True)
    rep = {"case": name, "mode": mode, "trainer": str(run_job.trainer_path(C).relative_to(PROJECT)),
           "device": "cpu (AGH_DEVICE=cpu)" if os.environ.get("AGH_DEVICE", "").lower() == "cpu" else "GPU lease"}
    size, err = sandbox_size(R, "000", out)
    if size is None:
        return rep | {"ok": False, "stage": "size check", "error": err}
    rep["base_size"] = size
    for k, lim in (("params", budget.get("max_params")), ("flops", budget.get("max_flops"))):
        if lim and size[k] > lim:
            return rep | {"ok": False, "stage": "size check", "error": f"base {k} {size[k]} > limit {lim}"}
    n = int(samples or min(budget["train_samples"], 4096))
    t0 = time.time()
    r = run_job.run(C, R / "cands" / "000", out, n, "val", 0, "{}", timeout_s=3600)
    wall = time.time() - t0
    if r.get("failed"):
        return rep | {"ok": False, "stage": "training", "log_tail": Path(r["log"]).read_text()[-3000:]}
    tt = float(r.get("train_time_s") or wall)
    per = budget["train_samples"] / max(1, n)
    return rep | {"ok": True, "smoke_train_samples": n, "smoke_score": r.get("score"), "smoke_wall_s": round(wall, 1),
                  "estimated_train_s_per_candidate": round(tt * per, 0),
                  "estimated_hours_per_run": round((tt * per + max(0.0, wall - tt) + 240) * budget["candidates_per_run"] / 3600, 1),
                  "note": "estimate = train time scaled to train_samples + eval/load + ~4 min agent time, per candidate"}


def rerun_best(case, run, split="val"):
    """Retrain the selected model with the run's frozen trainer and final budget (reproducibility check).
    Default split val: the test split was graded once by finalize and is not re-graded here."""
    R = rdir(case, run)
    B, fin = jload(R / "boundary.json"), jload(R / "final.json")
    if not fin:
        raise SystemExit("run not finalized yet")
    if split == "test":
        print("note: this evaluates the test split again; the run's graded S_test in final.json does not change",
              file=sys.stderr)
    f = B.get("final") or {}
    out = R / "best" / f"rerun_{split}_{time.strftime('%Y%m%d-%H%M%S')}"
    res = run_job.run(PROJECT / "cases" / case, R / "cands" / fin["best"], out,
                      f.get("train_samples", B["train_samples"]), split, 0, "{}", timeout_s=f.get("timeout_s", B["timeout_s"]),
                      extra_args=f.get("extra_args", []), trainer=B.get("trainer_path"), loader=B.get("loader_path"))
    return {"best": fin["best"], "split": split, "score": res.get("score"), "graded_S_test": fin.get("S_test"),
            "out": str(out.relative_to(PROJECT)), "failed": res.get("failed", False)}


# ---------------------------------------------------------------- status
def status(case=None, run=None):
    if case and run:
        R = rdir(case, run)
        st = jload(R / "state.json")
        toks = [json.loads(l) for l in open(R / "tokens.jsonl")] if (R / "tokens.jsonl").exists() else []
        sc = scored(R)
        return {"run": f"{case}/{run}", "candidates": len(cand_ids(R)) - 1, "scored": len(sc),
                "best": max(sc, key=lambda s: s[1])[:2] if sc else None, "spaces": st["spaces"],
                "bandit": st["bandit"], "agent_calls": len(toks),
                "cost_usd": round(sum(t.get("cost_usd") or 0 for t in toks), 2),
                "final": jload(R / "final.json"), "judge": jload(R / "judge.json")}
    runs = sorted(str(p.relative_to(PROJECT / "runs")) for p in (PROJECT / "runs").glob("*/*") if p.is_dir()) \
        if (PROJECT / "runs").exists() else []
    return {"runs": runs}
