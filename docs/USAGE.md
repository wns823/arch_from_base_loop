# ag_harness usage reference

Details behind the [README](../README.md).

## Case files and contract

    cases/<name>/
      task.md              what agents read: the task, the change you want, and an exact "Runner contract"
      base/model.py        your model: class Model(nn.Module); module-level CONDITIONAL; UPPER_CASE constants = hyperparameters
      base/train_config.py BATCH_SIZE, loss_fn(pred, y), make_optimizer(params), optional make_scheduler(opt, total_steps)
      data_loader.py       fixed (agents cannot edit it)
      trainer.py           generic training/scoring loop from the template; edit only for unusual training
      budget.yaml          budget and target (table below)
      data/                optional; may be a symlink to another disk (bound read-only into the sandbox)

Agents may write only `model.py` and `train_config.py` (one file each) and `notes.json`.

**`data_loader.py` functions** (called by the template's `trainer.py`)

| Function | Returns |
|---|---|
| `load(split)` with split = train, val or test | `(xs, y)`: a tuple of input tensors sharing the first dimension, and the targets |
| `model_args(xs, conditional)` | Positional args for `model(...)`. Return `None` for an input the base model should not get |
| `score(pred, y)` | The selection score as a float. Higher is better unless `higher_is_better: false` |
| `example_inputs(conditional)` | Args for one sample, used for the size check |
| `model_init_kwargs()` (optional) | Keyword args for `Model(...)` |
| `EVAL_BATCH` (optional, default 1) | Samples per model call at evaluation. Keep it at 1 so a model cannot use statistics across test samples |

**Trainer contract** (only if you write your own `trainer.py`)
- It runs in the sandbox with `/case` (your case folder, read-only), `/cand` (the candidate, read-only) and `/out`, with no network.
- It receives `--train_samples N --eval val|test --seed S --model_kwargs JSON`, plus `final_extra_args` at finalize.
- It must write `/out/result.json` with a finite `score`. Include `train_time_s`, which the harness uses for timeouts.
- A non-zero exit or a missing `result.json` counts as a failed candidate. The last 3,000 characters of the log go back to the agent.
- Only packages from `.venv` are available.

## budget.yaml

Only `train_samples` is required.

| Key | Default | Meaning |
|---|---|---|
| `train_samples` | required | Training samples seen per candidate. This is the search budget, and every candidate gets the same one |
| `candidates_per_run` | 30 | Models tried per run. Use at least 5 |
| `target` | none | `{value, metric, higher_is_better}`. Agents see the number. `final.json` reports `ratio_to_target` and a verdict (≥ 0.95 × target) |
| `higher_is_better` | from `target`, else true | For an error metric, set it to false. The loop then maximises −score, and LEDGER shows negated scores |
| `sigma_seed` | `auto` | Run-to-run noise. `auto` trains the base model with seeds 0, 1 and 2 at `init-run` and uses their s.d. Rewards and stop rules are measured in units of σ |
| `max_params`, `max_flops` | `size_factor` × base | Size limits. A candidate over either limit is rejected |
| `size_factor` | 4 | Multiplier used when the limits are not given |
| `timeout_ratio_to_H_time` | 3 | Candidate timeout = this × the base model's training time + 300 s |
| `notes_for_agents` | none | A list of constraints or ideas shown to every agent ("keep the encoder", "must stay causal") |
| `final_train_samples` | = `train_samples` | Retrain the selected model with a longer budget before the single test score |
| `final_extra_args` | none | Extra trainer flags for the final training |
| `reselect_top3` | false | Retrain the top 3 with a second seed and pick the best mean |
| `mode` | `improve` | `reproduce` turns on the paper-reproduction rules: date-capped web search, leak check, and required gate fields |
| `trainer` | `trainer.py` | Trainer file name, in `cases/<name>/` or `harness/` |
| `ssa_protocol`, `reward_mode` | 2, `delta` | Search-space format and bandit reward. Keep the defaults |

## Commands

| Command | What it does |
|---|---|
| `agh new-case <name>` | Create a case from the template |
| `agh check-case <name> [samples]` | Budget check, sandboxed size check and a short training of the base model. Prints a time estimate |
| `agh init-run <case> <run>` | Create `runs/<case>/<run>/`, freeze the inputs, and train the base model as candidate 000 |
| `agh loop <case> <run>` | Search, then finalize. If it dies, run it again to resume |
| `agh status [<case> <run>]` | Progress summary |
| `agh next <case> <run>` | Show the next decision (dry run) |
| `agh rerun-best <case> <run> [val\|test]` | Retrain the selected model with the run's frozen trainer. Default val; the graded S_test never changes |
| `agh finalize <case> <run>` | Run automatically at the end of `loop`. Refuses a second call |
| `agh doctor` | Check the server |
| `harness/bin/chain.sh <case> [policy] [n] [prefix]` | Run n independent runs of one case back to back. Log in `reports/` |

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `AGH_DEVICE` | GPU | `cpu` = skip the GPU lease and train on CPU |
| `AGH_GPU` | `0` | nvidia-smi index of the GPU to lease |
| `AGH_MODEL` | `opus` | Claude model for the agents |
| `AGH_POLICY` | `ucb` | Who to call next: `ucb`, `alternate`, `exploiter` |
| `AGH_WEAVE`, `AGH_WEAVE_PY` | on, `~/miniforge3/bin/python` | Mirror agent calls to W&B Weave using a separate Python with `weave`. Skipped if that Python is missing |
| `AGH_PYTHON` | `.venv/bin/python` | Python used by `agh` and inside the sandbox |

## Run folder

| File | Contents |
|---|---|
| `final.json` | Selected model, S_val, S_test, ratio to target, verdict |
| `best/` | Selected model's code, the frozen trainer and loader, the final training log, and the rerun command |
| `LEDGER.md` | One row per candidate: agent, parent, changed modules and hyperparameters, idea, S_val, status |
| `cands/<id>/` | Each candidate's code, `notes.json` (what changed and why) and `result.json` |
| `results/<id>/job.log` | Training log of each candidate |
| `frozen/` | Trainer, data loader, budget and task as they were at `init-run`, with sha256 in `MANIFEST.json` |
| `space_v<k>.json` | Search-space versions |
| `events.jsonl` | Every decision, reward and revision |
| `calls/<seq>_<role>.json` | Full input and output of each agent call, with every tool call |
| `memory/<role>.md` | The summary the agent got on its latest call (spec: `.claude/agent-memory/`) |
| `steer.md` | Your instructions for the agents (optional) |

## Rules the code enforces

- **Run length:** at most `candidates_per_run` candidates. A run stops early after two search-space revisions in a row bring no gain, and allows at most five revisions.
- **Bandit:** UCB over explorer and exploiter with reward clip((score − best)/σ, −1, +3); a failed candidate gets −2 (exploiter) or −1 (explorer). When an exploiter candidate gains, the explorer candidate whose module it tuned (nearest explorer among its last 3 parents) gets half of that reward as extra credit. Each agent gets at least 2 of every 8 candidates.
- **Sandbox:** bubblewrap, with no network and one GPU. A size check runs before training.
- **Access:** agents read only their own run folder and write only the candidate files. There is no memory across runs.
- **Test split:** scored once per run, by `finalize`.

## Install details

- **Requirements:** Linux; an NVIDIA GPU or `AGH_DEVICE=cpu`; `bwrap` with unprivileged user namespaces; Python ≥ 3.10; Claude Code logged in.
- **torch build:** if the default wheel does not match your CUDA driver, use `TORCH_INDEX=https://download.pytorch.org/whl/cu121 harness/bin/setup.sh`.
- **Copying case data between servers:** `harness/bin/sync_data.sh you@server:/path/to/ag_harness <case> [--to /big/disk]` (rsync; a symlinked data folder goes to the disk you name).

## Glossary

| Term | Meaning |
|---|---|
| A0 | The base model (candidate 000) |
| H | Reference model reimplemented by hand for a validation case. Agents never see it |
| gate | Calibration of a validation case before any run: H vs A0, budget, size limits |
| σ | Run-to-run noise of the score (`sigma_seed`) |
| SSA | The search-space agent |
| coverage | Fraction of the space's module choices already tried. High coverage triggers a space revision |
