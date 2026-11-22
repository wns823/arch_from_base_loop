# Case template

`harness/bin/agh new-case <name>` copies this folder to `cases/<name>/`. Then:

1. `data_loader.py`: replace the toy data with yours; keep `load`, `model_args`, `score`, `example_inputs`.
2. `base/model.py`, `base/train_config.py`: your current model and training recipe.
3. `task.md`: describe the requirement and keep the Runner contract exact.
4. `budget.yaml`: `train_samples`, `target`, `candidates_per_run`, optional `notes_for_agents`.
5. `harness/bin/agh check-case <name>`: size check + a short sandboxed training of the base model.
6. `harness/bin/agh init-run <name> r1 && harness/bin/agh loop <name> r1`.

`trainer.py` is generic; edit it only if your training loop is unusual (multi-task losses, special batching).
