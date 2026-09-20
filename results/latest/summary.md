# Scheduler boundary comparison

Main commit: `3853733`. PR commit: `14b798c`. Python: `3.12.14`.

| Variant | Contract tests | Upstream cross-attention tests |
| --- | --- | --- |
| Main (`<`) | 1 failed, 3 passed: MM item split | not run here |
| PR #52412 (`<=`) | 1 failed, 3 passed: encoder-decoder blocked | 2 failed |
| Scoped candidate (`<=` only outside encoder-decoder) | 4 passed | 2 passed |

Each cell is backed by the corresponding `.log` in this directory. These are scheduler correctness checks, not speed or model-quality metrics.
The script restored the PR worktree's original file bytes after testing.
