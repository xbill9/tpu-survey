Excerpt of /home/xbill/gemma4-dev/tpu-vllm-v5e1-2b-w8a8emb4/README.md lines 1-12, copied 2026-10-25

----- lines 1-12 -----
1: # tpu-vllm-v5e1-2b-w8a8emb4
2: 
3: Results for one cell of the Gemma 4 repack sweeps: **`/work/models/gemma-4-E2B-it-qat-w8a8-int8-emb4`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.
4: 
5: Slot 5, `w8a8emb4`: `w8a8` with `embed_tokens` (and the per-layer embeddings, where present) and an untied `lm_head` int4.
6: 
7: | Run | Checkpoint | Suite accuracy | Sweep |
8: |---|---|---:|---|
9: | [`2026-09-29-e4a-e2b-w8a8-emb4-v5e1`](benchmarks/runs/2026-09-29-e4a-e2b-w8a8-emb4-v5e1/) | `/work/models/gemma-4-E2B-it-qat-w8a8-int8-emb4` | 0.679 (3,880 records) | `jev-tpu-v5e1` |
10: | [`2026-09-29-w8e4-e2b-w8a8-emb4-v5e1`](benchmarks/runs/2026-09-29-w8e4-e2b-w8a8-emb4-v5e1/) | `/work/models/gemma-4-E2B-it-qat-w8a8-int8-emb4` | 0.677 (3,880 records) | `jev-tpu-v5e1` |
11: | [`2026-09-30-fillB-e2b-w8a8-emb4-hf-v5e1`](benchmarks/runs/2026-09-30-fillB-e2b-w8a8-emb4-hf-v5e1/) | `xbill9/gemma-4-E2B-it-qat-w8a8-ct-text-emb4` | 0.673 (3,880 records) | `jev-tpu-v5e1` |
12: 
