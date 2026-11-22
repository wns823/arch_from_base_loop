# Explorer memory

The loop writes `runs/<case>/<run>/memory/explorer.md` before every explorer call. The explorer swaps in a module, so its memory is about how modules have done in this run.

## Contents

1. **User instructions**, if `steer.md` exists.
2. **Run status:** the best candidate, its S_val, σ, and the gap to the target in σ.
3. **Your target:** the (axis, choice) the allocator picked, or off-axis, and the parent's module choices.
4. **Module gains after tuning:** for every (axis, choice) targeted so far, how many times it was tried and its mean gain in σ. The gain is the best score of the explorer candidate and its tuned descendants, minus the parent's score. Modules are judged after tuning, not on their first, untuned score.
5. **Failed explorer candidates:** the module tried and the last error line, so the same mistake is not repeated.
6. **Earlier tries of the target:** candidates that already used this choice, with their scores.
7. **Off-axis modules already tried.**

## Source

`cands/*/notes.json` and `result.json`, gains from `choice_gains`, module choices from `eff_axes`, and σ and the target from `state.json` and `boundary.json`.
