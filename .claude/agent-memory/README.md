# Agent memory

How each agent's past record is kept and given back to it. One folder per agent:

| Agent | Spec | Written each call to |
|---|---|---|
| exploiter | [exploiter/MEMORY.md](exploiter/MEMORY.md) | `runs/<case>/<run>/memory/exploiter.md` |
| explorer | [explorer/MEMORY.md](explorer/MEMORY.md) | `runs/<case>/<run>/memory/explorer.md` |
| search-space | [search-space/MEMORY.md](search-space/MEMORY.md) | `runs/<case>/<run>/memory/search-space.md` |

## How it works

1. **The record.** Every agent call leaves files in its run folder:
   - candidates: `cands/<id>/` with code, `notes.json` (what changed and why) and `result.json` (score or error)
   - search-space versions: `space_v<k>.json`
   - decisions and rewards: `events.jsonl`
   - the full input and output of the call: `calls/<seq>_<role>.json`
   This record is the agents' long-term memory.
2. **The summary.** Before each agent call, the loop code (`harness/driver.py: write_digest`, not an LLM) reads the record and writes a short summary. The summary holds only what that agent needs for this call (spec in `<role>/MEMORY.md`).
3. **The call.** The call message tells the agent to read the summary first. The agent may open the full `LEDGER.md` and the candidates' files if it needs more.

## Rules

- **Code writes the memory; agents only read it.** Agents cannot write `memory/` or `LEDGER.md`. A summary written by code cannot drift, flatter the agent or leak an answer.
- **One run, one memory.** An agent sees only the record of its own run (`$AGH_RUN_DIR`, enforced by `.claude/hooks/policy.py`). Runs of the same case stay independent, so one run cannot copy another's result.
- **User instructions come first.** If the user writes `runs/<case>/<run>/steer.md`, its text is put at the top of every summary from the next call on.
- **Nothing from outside the run.** No answer keys, no test scores, no other runs.

The Claude Code `memory:` field is deliberately not set in `.claude/agents/*.md`. That field gives an agent a shared notebook that it writes itself and that carries over between runs. Both break the rules above. That notebook was used until 2026-11-20 and is archived outside the repo.

To change what an agent remembers, edit `write_digest` in `harness/driver.py` and the matching `MEMORY.md` together.
