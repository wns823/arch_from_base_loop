#!/usr/bin/env python3
"""PreToolUse policy for ag_harness.

    policy.py guard [--role R]     PreToolUse
    policy.py log   [--role R]     PostToolUse (SSA web calls → $AGH_RUN_DIR/search_log.jsonl)

Roles: search-space, exploiter, explorer, judge (from --role, env AGH_ROLE set by the loop, or the hook input's
agent_type); none = main session.
- Every harness role: never touches ~/ag_harness_private, validation/, project_spec.md, spec_interview/.
- exploiter/explorer: read only inside their own run runs/<case>/<run>/ ($AGH_RUN_DIR), write only inside cands/<id>/,
  no shell, no web (REQ-016: the loop code trains and scores).
- search-space: read inside runs/<case>/<run>/, write only space_v*.json there, no shell,
  web allowed (date cap is checked post hoc by the judge from the search log).
- judge: retired 2026-11-20 (grading is code); every call under this role is denied.
- main session: unrestricted here (setup work); global deny rules live in settings.json.
Fails closed for harness roles, open for the main session.
"""
import json
import os
import re
import time
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2]).resolve()
PRIVATE = (Path.home() / "ag_harness_private").resolve()
ROLES = {"search-space", "exploiter", "explorer"}
FORBIDDEN = [PRIVATE, PROJECT / "validation", PROJECT / "project_spec.md", PROJECT / "spec_interview",
             PROJECT / "CLAUDE.md"]


def deny(msg):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": f"[ag_harness policy] {msg}"}}))
    return 0


def under(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def paths_in(tool, inp):
    keys = {"Read": ["file_path"], "Write": ["file_path"], "Edit": ["file_path"], "Glob": ["path"], "Grep": ["path"]}
    out = []
    for k in keys.get(tool, []):
        v = inp.get(k)
        if v:
            out.append(Path(v if os.path.isabs(v) else PROJECT / v))
    if tool in ("Glob", "Grep") and not inp.get("path"):
        out.append(Path(os.getcwd()))
    return out


def run_dir_of(p: Path):
    m = re.match(r"^(runs/[^/]+/[^/]+)", str(p.resolve().relative_to(PROJECT))) if under(p, PROJECT) else None
    return PROJECT / m.group(1) if m else None


def guard_judge(tool, inp):
    """judge agent retired 2026-11-20 (grading is code only): every tool call under this role is denied."""
    return deny("the judge agent was removed; grading is done by harness/judge.py (code)")
    if tool not in ("Read", "Glob", "Grep", "Write"):
        return deny(f"judge may not use {tool}")
    for p in paths_in(tool, inp):
        if tool == "Write":
            if not (run_dir_of(p) and p.name == "judge_items.json" and p.parent == run_dir_of(p)):
                return deny("judge may only write runs/<case>/<run>/judge_items.json")
        elif not (under(p, PRIVATE / "answer_keys") or run_dir_of(p)):
            return deny("judge may read only answer keys and runs/")
    return 0


def log_search(data, role):
    """PostToolUse: append every SSA web call to $AGH_RUN_DIR/search_log.jsonl (judge checks the date cap)."""
    run_dir = os.environ.get("AGH_RUN_DIR")
    if role != "search-space" or not run_dir or data.get("tool_name") not in ("WebSearch", "WebFetch"):
        return 0
    resp = data.get("tool_response")
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": data.get("tool_name"), "input": data.get("tool_input"),
           "response_head": (json.dumps(resp) if not isinstance(resp, str) else resp)[:4000]}
    with open(Path(run_dir) / "search_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return 0


def guard(data, role):
    tool = data.get("tool_name", "")
    inp = data.get("tool_input", {}) or {}
    if role == "judge":
        return guard_judge(tool, inp)
    if role not in ROLES:
        return 0
    text = json.dumps(inp)
    if str(PRIVATE) in text or "ag_harness_private" in text:
        return deny("the private answer-key root is off limits to harness agents")
    for p in paths_in(tool, inp):
        if any(under(p, f) or p.resolve() == f.resolve() for f in FORBIDDEN):
            return deny(f"{p} is outside what {role} may see")
        if run_dir_of(p) is None:
            return deny(f"{role} may only access runs/<case>/<run>/ (got {p})")
        if tool in ("Glob", "Grep") and p.resolve() == run_dir_of(p).resolve():
            # a search rooted at the run folder would also see frozen/ and best/
            return deny(f"{role}: search inside a subfolder (cands/, workspace/, memory/) or give a file path; "
                        "a search over the whole run folder is not allowed")
        rel_run = str(p.resolve().relative_to(run_dir_of(p)))
        if rel_run == ".":
            rel_run = ""
        if rel_run.split("/")[0] in ("frozen", "best"):
            # fix 2026-11-22: frozen/ and best/ hold the trainer and budget.yaml, whose comments can name the source
            # paper (validation cases); agents use workspace/ (task.md, data_loader.py, base/) instead
            return deny(f"{role} may not read {rel_run.split('/')[0]}/ (trainer and budget are fixed and not part of the task)")
        own = os.environ.get("AGH_RUN_DIR")   # fix 2026-11-22: only this call's own run, never another run
        if own and run_dir_of(p).resolve() != Path(own).resolve():
            return deny(f"{role} may only access its own run {own} (got {p})")
        rel = str(p.resolve().relative_to(PROJECT))
        if tool in ("Write", "Edit"):
            if role == "search-space" and not re.match(r"^runs/[^/]+/[^/]+/space_v\d+\.json$", rel):
                return deny("search-space may only write runs/<case>/<run>/space_v<k>.json")
            if role in ("exploiter", "explorer") and not re.match(r"^runs/[^/]+/[^/]+/cands/[^/]+/", rel):
                return deny(f"{role} may only write inside runs/<case>/<run>/cands/<id>/")
    if tool in ("WebSearch", "WebFetch") and role != "search-space":
        return deny(f"{role} has no internet access")
    if tool == "Bash":
        return deny(f"{role} has no shell; the loop code runs everything (REQ-016)")
    return 0


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        data = {}
    role = (sys.argv[sys.argv.index("--role") + 1] if "--role" in sys.argv[:-1]
            else os.environ.get("AGH_ROLE") or data.get("agent_type") or "main")
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "log":
            return log_search(data, role)
        return guard(data, role)
    except Exception as e:  # fail closed for harness roles
        if role in ROLES:
            return deny(f"policy error: {e}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
