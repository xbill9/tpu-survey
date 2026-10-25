Excerpt of /home/xbill/gemma4-dev/tpu-vllm-v5e4-12b-w8a8emb4/README.md lines 21-39, copied 2026-10-25

----- lines 21-39 -----
21: ### Measured on v5e-4
22: 
23: On 2026-10-02 this rig provisioned (flex-start `v5litepod-4` in `us-west4-a`, capacity within 5 minutes)
24: and booted through its own `create_tpu_queued_resource` at `TENSOR_PARALLEL_SIZE=4`
25: ([`benchmarks/runs/2026-10-02-rig-boot-v5e4`](benchmarks/runs/2026-10-02-rig-boot-v5e4)). The patched int8 W8A8 linears, int4 embedding tables and int4 `lm_head`
26: all work at TP=4: chat and a tool call verified, ready about 17.5 minutes after the queued resource was
27: created (engine init 541 s, 501 s of it compilation).
28: 
29: | | v5e-1 sibling | v5e-4, TP=4 | v5e-4 / v5e-1 |
30: |---|---:|---:|---:|
31: | Weights per chip | 11.31 GiB | 3.25 GiB | |
32: | KV cache | 9,728 tokens | 122,624 tokens | 12.61x |
33: | Output tok/s, 1 request | 57.0 | 128.4 | 2.25x |
34: | Output tok/s, 4 requests | 215.7 | 450.9 | 2.09x |
35: | Output tok/s, 16 requests | 713.4 | 1,212.9 | 1.70x |
36: 
37: Both columns are each rig's own boot run with the same load client (256 output tokens, greedy, median of
38: three passes). Predicted ~122,400 KV tokens and 3.22 GiB per chip; measured 122,624 and 3.25. At
39: `MAX_MODEL_LEN` 8192 the pool holds 14.97 requests of full length, so the context limit sits far below
