"""Report helpers used by driver.finalize (reproduced, leak_check). `agh judge` only re-prints them.

Grade a finalized run with code only (user decision 2026-11-20: the judge agent was removed).

Pass criterion (one training run): S_test ≥ 0.95 × target (paper's number, same data/split/metric) → REPRODUCED.
                For a lower-is-better metric: target / S_test ≥ 0.95. The same ratio for the gate's H checks the setup.
G          = (S_test − S_A0) / (S_H − S_A0), with S_H, S_A0 the gate's test means (private verdict.json)
leak check = SSA web calls vs the case's search date cap and the answer paper's identifiers
No LLM reads an answer key. Item-level rediscovery (R, space recall) is no longer computed.
"""
import json
import re
from pathlib import Path

from harness import driver

PRIVATE = Path.home() / "ag_harness_private"


def leak_check(R, case):
    B = driver.jload(R / "boundary.json")
    pats = driver.jload(PRIVATE / "answer_keys" / "leak_patterns.json", {}).get(case, [])
    hits, late = [], []
    if (R / "search_log.jsonl").exists():
        for l in open(R / "search_log.jsonl"):
            r = json.loads(l)
            blob = json.dumps(r).lower()
            if any(p.lower() in blob for p in pats):
                hits.append(r)
    for k in driver.jload(R / "state.json")["spaces"]:
        for a, v in driver.jload(R / f"space_v{k}.json")["axes"].items():
            for s in v.get("sources", []):
                m = re.search(r"\((\d{4}-\d{2}-\d{2})\)\s*$", s)
                if m and m.group(1) >= B["search_date_cap"]:
                    late.append(s)
    return {"answer_paper_hits": hits, "sources_after_cap": late, "leak": bool(hits or late)}


def judge(case, run):
    R = driver.rdir(case, run)
    final = driver.jload(R / "final.json")
    if final is None:
        return {"error": "finalize first"}
    if driver.jload(R / "judge.json"):
        return {"error": "already graded", "judge": driver.jload(R / "judge.json")}
    B = driver.jload(R / "boundary.json")
    tgt = B.get("target")
    s = final.get("S_test")
    v = driver.jload(PRIVATE / "gate" / (case[:-3] if case.endswith("neg") else case) / "verdict.json")
    vv = v[case] if v and case in v else None
    G = None
    if vv and s is not None:
        G = (s - vv["A0_test"]) / (vv["H_test"] - vv["A0_test"])
    res = {"case": case, "run": run, "best": final.get("best"), "S_val": final.get("S_val"), "S_test": s,
           "target": tgt["value"] if tgt else None,
           "A0_test": vv["A0_test"] if vv else None, "H_test": vv["H_test"] if vv else None,
           "gap_to_target": (s - tgt["value"]) if (tgt and s is not None) else None,
           "G": G, "R": None, "leak": leak_check(R, case)}
    if tgt:
        hib = bool(tgt.get("higher_is_better", True))
        res["ratio_to_target"], res["verdict"] = reproduced(s, tgt["value"], hib)
        if vv:   # same check on the gate's H: tells a setup gap from a search gap
            res["setup_check_H_ratio"], res["setup_check_H"] = reproduced(vv["H_test"], tgt["value"], hib)
    driver.jsave(R / "judge.json", res)
    return res


# ---------------------------------------------------------------- reproduction criterion (user decision 2026-11-20)
REPRO_FRAC = 0.95   # reproduced iff the final model reaches 95% of the paper's number (one training run)


def reproduced(S, T, higher_is_better=True):
    """ratio = S/T (higher is better) or T/S (lower is better); ratio ≥ 0.95 → REPRODUCED."""
    if S is None or not T or (not higher_is_better and not S):
        return None, None
    ratio = S / T if higher_is_better else T / S
    return ratio, ("REPRODUCED" if ratio >= REPRO_FRAC else "NOT REPRODUCED")
