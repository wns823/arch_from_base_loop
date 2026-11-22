# Task

<!-- What agents read. Plain words, no paper names. Keep the "Runner contract" section exact: every mismatch costs a
failed candidate, because agents cannot run code themselves. -->

You are given a sequence-to-scalar model (`base/model.py`). It reads a one-hot sequence `seq [b, 4, 64]` and predicts a
scalar response per sequence.

**New requirement:** each sample also has an 8-dimensional context vector `ctx [b, 8]`, and the response depends on it.
Change the model so that it uses the context, including contexts never seen in training (validation and test contexts
come from held-out clusters).

## Runner contract

- You may edit `model.py` and `train_config.py` only (one file each; no other files).
- The runner builds `Model(ctx_dim=8)` and calls `model(seq, ctx)` with `seq [b, 4, 64]`, `ctx [b, 8]`, expecting `[b]`.
  Set `CONDITIONAL = True` in `model.py` to receive `ctx`; with `CONDITIONAL = False` it passes `None`.
- Training: random minibatches of `BATCH_SIZE` until a fixed number of training samples has been seen (set by the
  runner); loss `loss_fn(pred, y)`, optimizer `make_optimizer(params)`, optional `make_scheduler(opt, total_steps)`.
- Evaluation calls the model with one sample at a time (b = 1).
- Model size limits (parameters, forward FLOPs per sample) are enforced; oversized candidates are rejected.
- Selection score: R² of the predictions on the validation samples (higher is better).
- Final evaluation: the selected model is retrained once and scored once on the test samples.
