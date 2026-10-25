Excerpt of /home/xbill/gemma4-dev/tpu-vllm-v5e1-26b-q4w4a16/README.md lines 1-19, copied 2026-10-25

----- lines 1-19 -----
1: # tpu-vllm-v5e1-26b-q4w4a16
2: 
3: Results for one cell of the Gemma 4 repack sweeps: **`/work/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.
4: 
5: Slot 5, `q4w4a16`: W4A16 holding the QAT Q4_0 grid exactly (xbill9 repack of `-qat-q4_0-unquantized`).
6: 
7: | Run | Checkpoint | Suite accuracy | Sweep |
8: |---|---|---:|---|
9: | [`2026-09-28-lowmem7-26b-lowmem-v5e1`](benchmarks/runs/2026-09-28-lowmem7-26b-lowmem-v5e1/) | `/work/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct` | 0.754 (3,880 records) | `jev-tpu-v5e1` |
10: 
11: Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
12: `quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
13: lists; they cover two cells, so they stay with the sweep.
14: 
15: ## Other runs
16: 
17: No suite; see each run's `README.md`.
18: 
19: - [`2026-09-30-fillD-26b-repack-v5e1`](benchmarks/runs/2026-09-30-fillD-26b-repack-v5e1/): output tok/s at 1 / 4 / 16: 31 / 102 / 201
