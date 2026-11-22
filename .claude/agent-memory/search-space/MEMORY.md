# Search-space agent memory

The loop writes `runs/<case>/<run>/memory/search-space.md` before every search-space call (the first space and each revision). The search-space agent decides which modules to search, so its memory is about what earlier spaces achieved.

## Contents

1. **User instructions**, if `steer.md` exists.
2. **Run status:** the best candidate, its S_val, σ, and the gap to the target in σ. The call message adds a direction: WIDEN, KEEP or NARROW.
3. **Space revisions so far:** each version with why it was called, the best score when it started, and whether it improved the best by at least 0.5σ.
4. **Per module choice:** for every (axis, choice) in the run, the number of candidates trained and failed, the best S_val, and the mean gain after tuning.
5. **Top 5 models** with their module choices.
6. **Modules explorers found missing:** the `space_gap` notes that explorers wrote when they thought the space lacked a module.

## Source

`state.json` (`ssa_log`), `space_v*.json`, `cands/*/notes.json` and `result.json`, and `choice_gains`.
