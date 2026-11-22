"""agh: run driver and utilities (REQ-016). Called by the main session; agents have no shell. Each prints JSON.

  agh init-run <case> [run]       create runs/<case>/<run>/, train base as candidate 000
  agh loop <case> <run>           allocator → agent → automatic check/train/S_val … → finalize (test once)
  agh next <case> <run>           show the allocator's next decision (dry run)
  agh run-candidate <case> <run> <id>   (re)train one candidate (main session repair only)
  agh ledger <case> <run>         regenerate LEDGER.md
  agh space-check <case> <run> <k>      validate space_v<k>.json
  agh finalize <case> <run>       best S_val → S_test once (refuses a second call)
  agh judge <case> <run>          grade (code only): S_test vs target, G, search-date leak check
  agh status [<case> <run>]
  agh doctor                      check this server: python deps, claude CLI, GPU, bwrap sandbox, case data
  agh new-case <name>             copy cases/_template to cases/<name>/ (generic trainer, toy data, budget defaults)
  agh check-case <name> [samples] budget check + sandboxed size check + short base training (no agents); time estimate
  agh rerun-best <case> <run> [val|test]   retrain the selected model with the run's frozen trainer (default val)
"""
import json
import sys

from harness import driver


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, a = argv[0], argv[1:]
    if cmd == "doctor":
        from harness import doctor
        return 0 if doctor.main() else 1
    need = {"init-run": 1, "loop": 2, "next": 2, "run-candidate": 3, "ledger": 2, "space-check": 3, "finalize": 2,
            "judge": 2, "new-case": 1, "check-case": 1, "rerun-best": 2}
    if cmd in need and len(a) < need[cmd]:
        print(__doc__)
        return 2
    if cmd in ("loop", "next", "run-candidate", "ledger", "space-check", "finalize", "judge", "rerun-best") \
            and not (driver.rdir(a[0], a[1]) / "state.json").exists():
        print(f"run {a[0]}/{a[1]} not found: start it with  harness/bin/agh init-run {a[0]} {a[1]}", file=sys.stderr)
        return 2
    if cmd == "new-case":
        out = driver.new_case(a[0])
        print(json.dumps(out, indent=1)); return 0
    if cmd == "check-case":
        out = driver.check_case(a[0], int(a[1]) if len(a) > 1 else None)
        print(json.dumps(out, indent=1, default=str)); return 0 if out.get("ok") else 1
    if cmd == "rerun-best":
        out = driver.rerun_best(a[0], a[1], a[2] if len(a) > 2 else "val")
        print(json.dumps(out, indent=1, default=str)); return 0
    if cmd == "init-run":
        out = driver.init_run(a[0], a[1] if len(a) > 1 else None)
    elif cmd == "loop":
        out = driver.loop(a[0], a[1])
    elif cmd == "next":
        out = driver.allocate(driver.rdir(a[0], a[1]))
    elif cmd == "run-candidate":
        out = driver.train_candidate(driver.rdir(a[0], a[1]), a[2])
        driver.regen_ledger(driver.rdir(a[0], a[1]))
    elif cmd == "ledger":
        driver.regen_ledger(driver.rdir(a[0], a[1]))
        out = {"ok": True}
    elif cmd == "space-check":
        out = {"error": driver.validate_space(driver.rdir(a[0], a[1]), int(a[2]))}
    elif cmd == "finalize":
        out = driver.finalize(a[0], a[1])
    elif cmd == "judge":
        from harness import judge
        out = judge.judge(a[0], a[1])
    elif cmd == "status":
        out = driver.status(*(a[:2] if len(a) >= 2 else []))
    else:
        print(__doc__)
        return 2
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
