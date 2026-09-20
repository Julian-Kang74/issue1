# Scheduler boundary case study

This project studies vLLM [issue #52306](https://github.com/vllm-project/vllm/issues/52306) and the existing proposed fix in [PR #52412](https://github.com/vllm-project/vllm/pull/52412). It records a reproducible comparison; it is not a new upstream PR.

## A multimodal item on one token axis

Suppose the first 400 request tokens are text, followed by a multimodal item at positions `[400, 1200)` with length 800. `num_computed_tokens` is the position already processed, and `num_new_tokens` is the count proposed for this scheduling step.

| Already computed | Step budget | End position without rollback | Expected scheduled count |
| ---: | ---: | ---: | ---: |
| 0 | 500 | 500, inside the multimodal item | 400: stop at the item's start |
| 400 | 500 | 900, inside the multimodal item | 0: wait until the whole item fits |
| 400 | 800 | 1200, exactly the item's end | 800 |

The original guard rolls back to avoid splitting an item only when `num_computed_tokens < start_pos`. In the first row, `0 < 400` holds, so scheduling stops at 400. In the second row, `400 < 400` is false, so the item is split. Changing `<` to `<=` gives the desired zero in the second row.

## Why encoder–decoder requests regress

For an encoder–decoder model, encoder input and decoder tokens are on separate position axes. The scheduler uses the same function and places the encoder input at `start_pos=0`. For example, take an encoder input of length 200 and a decoder prefill of 100 tokens: `num_computed_tokens=0`, `num_new_tokens=100`, `start_pos=0`, and `num_encoder_tokens=200`.

The PR's new condition sees `0 <= 0` and `100 < 200`, then incorrectly reduces the scheduled decoder count to zero. This does not mean the encoder input would be split into two chunks of 100. It means that token counts from separate axes were compared as though they described one contiguous sequence. In this branch, both upstream cross-attention tests fail because no request is scheduled.

The local candidate adds `and not self.is_encoder_decoder` to this particular no-split guard. Encoder–decoder requests still go through the function's later encoder cache and budget checks. The candidate passes this project's four behavioral checks and the two upstream tests, but that is not a full test suite or proof of model-output correctness.

## Reading the results

After running `scripts/compare.py`, see [`results/latest/summary.md`](../results/latest/summary.md) and its five adjacent logs. `main_contract.log` shows that the original code schedules 500 tokens at the 400/500 boundary. `pr_contract.log` shows the encoder–decoder request being blocked. `candidate_contract.log` shows all four direct checks passing. The two `official_cross_attn` logs contain the upstream cross-attention test results.

These results establish specific scheduler behavior. They are not a GPU benchmark and do not measure model-answer quality. The PR's LLaVA pytest could not load with the shared environment's incompatible `transformers` version, so the direct scheduler checks should not be described as a full integration test.
