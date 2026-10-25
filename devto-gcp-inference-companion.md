---
title: "Inference on Google Cloud: Precision, Memory and Capacity on TPU, from v5e to Ironwood"
published: false
description: "The companion to the talk: what the number format, memory bandwidth, slice shape and capacity rules do to Gemma 4 on Cloud TPU v5e and v6e, against the L4, Blackwell, MI300X and Inferentia, with the cost of each."
tags: googlecloud, tpu, machinelearning, gemma
cover_image: https://raw.githubusercontent.com/xbill9/tpu-survey/main/devto-cover.eef6abe3.jpg
---

This article is the written companion to the talk "Inference on Google Cloud". It covers serving Gemma 4 on Cloud TPU v5e and v6e through vLLM, measured in the gemma4-dev Cloud TPU rigs between July and October 2026, and compares them with the NVIDIA L4 on Google Cloud, Blackwell list prices, an AMD MI300X and AWS Inferentia. Four ideas explain most of the results: the number format the chip computes natively, the memory bandwidth that feeds it, the slice shape, and the rules that decide whether a TPU is there to rent at all.

https://github.com/xbill9/gemma4-dev

The hands-on steps run from a second repository, which starts one TPU and serves Gemma 4 on it with a single command:

https://github.com/xbill9/gcp-inference

The slides: https://docs.google.com/presentation/d/1hT8UXOEBueZfoFdIH0ZyhYhQ3WZJ_0xbCsvNDDs8H4A/edit

---

#### What Is the Problem?

This article compares four TPU generations, the L4 and Blackwell GPUs beside them, and every way Google Cloud prices a TPU chip-hour. The same Gemma 4 E2B serves at 144 tokens a second on a v5e in bf16 and at 243 in int8 with 4-bit embedding tables, on the same chip. A v6e-1 serves 3.3 times the L4's tokens on the same 26B checkpoint. And a zone that reports quota for 1,536 chips can still return a stockout.

---

#### Which Generations?

| Chip | HBM | Peak bf16 | Native low precision | Max pod | On-demand |
|---|---:|---:|---|---:|---:|
| v5e | 16 GB | 197 TFLOPs | int8, 393 TOPs | 256 | $1.20/hr |
| v6e Trillium | 32 GB | 918 TFLOPs | int8, 1,836 TOPs | 256 | $2.70/hr |
| v5p | 95 GiB | 459 TFLOPs | int8; fp8 at bf16 rate | 8,960 | $4.20/hr |
| TPU7x Ironwood | 192 GiB | 2,307 TFLOPs | fp8, 4,614 TFLOPs | 9,216 | $12.00/hr |

Google's first TPU, in its datacenters since 2015, was an 8-bit inference chip with a 92 TOPS peak, and Google announced Ironwood on 2025-04-09 as "the first Google TPU for the age of inference". The rigs measured v5e and v6e. v5p and TPU7x rows come from Google's documentation only.

vLLM reaches the TPU through tpu-inference, a JAX backend. XLA compiles one program per input shape, which makes the compile the largest part of a cold start.

---

#### At This Point You Should Have…

- The `gcloud` CLI with its `alpha` and `beta` components, signed in with `gcloud auth login` and `gcloud auth application-default login`
- A project with Cloud TPU quota for one v5e chip in us-west4-a
- `python3` with `pip install 'google-adk[extensions]' google-genai mcp`
- A clone of `xbill9/gcp-inference`

---

#### Step 1 — One TPU, One Command

`demo.sh` asks for a TPU v5e-1 as a flex-start request in us-west4-a, waits for the node, opens an SSH tunnel to vLLM on the TPU's port 8000, waits for the model to load and compile, then runs a Google ADK agent against it. Flex-start stops the node by itself after 4 hours. On a TPU that is already up, `./demo.sh again` skips the first two steps:

```bash
./demo.sh again
```

```text
== [0m01s] 3/5  Connecting to it over SSH (the demo reaches vLLM on the TPU's port 8000 through a tunnel)
   [0m05s] SSH works
   [0m11s] tunnel open: localhost:8005 -> TPU port 8000
== [0m11s] 4/5  Starting Gemma on vLLM (first start downloads and compiles: several minutes)
   [0m30s] vLLM is answering
  20:21:25Z  PASS  tpu        chat 0.198s; tool run filters=['id >= 10'] quoted=8 3.71s
READY
== [0m37s] 5/5  The demo: Gemma on the TPU counts ids with the count_ids tool from the MCP server
Question: How many of the ids are 10 or more? Answer with the number.
Ids: [0, 2, 3, 20, 21, 22, 23, 10, 11, 12, 13]
True answer: 8  tools: mcp
  tpu        run 1/3  ok   quoted=8  filters=['id >= 10']  3.87s
  tpu        run 2/3  ok   quoted=8  filters=['id >= 10']  0.51s
  tpu        run 3/3  ok   quoted=8  filters=['id >= 10']  0.49s
backend     runs  correct  right-filter  errors  median s  filters sent
tpu            3    3/3             3/3       0      0.51  id >= 10
```

The agent is an ADK `LlmAgent` whose model is `LiteLlm` pointed at vLLM's OpenAI-compatible API, and an ADK `McpToolset` starts `mcp_server.py` over stdio and offers it one tool. The model never counts. It calls `count_ids` on that MCP server, which returns the exact count, and each run is graded on the filter it sent as well as the number it quoted. Gemma 4 E2B on the v5e sent `id >= 10` in all three runs.

The node serves `google/gemma-4-E2B-it` in bf16 with `vllm/vllm-tpu:v0.30.0`, one chip, no sharding.

---

#### Step 2 — What the 16 GB Chip Holds

vLLM logs the chip's memory as it starts. In order:

```text
Init worker | ... | hbm=[(0.0, 15.75)]GiB
Init model | hbm=[(8.97, 15.75)]GiB
Memory statistics | total_hbm_limit_gb=15.75GiB | total_hbm_limit_cap_gb=14.49GiB | total_hbm_used_gb=8.97GiB | total_hbm_avail_gb=5.52GiB
TPU KV cache size: 321,280 tokens, Maximum concurrency for 8,192 tokens per request: 39.22x
Init kv-cache | num_total_layers=15 | num_blocks=[1255, ...] | regular_attn_shape=(num_blocks, (256, 1, 2, 256)) | regular_attn_dtype=bfloat16 | hbm=[(14.51, 15.75)]Gb
```

| HBM on one v5e chip | GiB |
|---|---:|
| Weights, bf16 | 8.97 |
| vLLM's ceiling (92% of 15.75) | 14.49 |
| Left for the KV cache | 5.52 |
| In use after the KV cache is allocated | 14.51 |
| Never allocated | 1.24 |

Each of the 15 cached layers holds 1,255 blocks of 256 tokens, one KV head of dimension 256, in bf16. That is 1,024 bytes per token per layer, 15,360 bytes per token across all 15, and 321,280 token slots in all. The model's config lists 35 layers with `num_kv_shared_layers: 20`: 20 layers reuse another layer's keys and values, so the cache holds 15.

---

#### Step 3 — The Full 131,072-Token Context

Gemma 4 E2B's config allows 131,072 positions. vLLM sizes its KV cache from the memory left after the weights, so raising `--max-model-len` from 8,192 to 131,072 changes the concurrency and leaves the cache the same size. To change a flag on a running TPU, replace the container:

```bash
gcloud compute tpus tpu-vm ssh gcp-inference-tpu-node --project <tpu-project> --zone us-west4-a --command \
  'sudo docker rm -f vllm && sudo docker run -d --name vllm --restart unless-stopped --privileged --net=host \
   -v /dev/shm:/dev/shm --shm-size 10gb -e HF_HOME=/dev/shm vllm/vllm-tpu:v0.30.0 \
   vllm serve google/gemma-4-E2B-it --served-model-name gemma --max-model-len 131072 --port 8000 \
   --max-num-batched-tokens 4096 --enable-auto-tool-choice --tool-call-parser gemma4'
```

```text
Memory statistics | total_hbm_limit_gb=15.75GiB | total_hbm_limit_cap_gb=14.49GiB | total_hbm_used_gb=8.97GiB | total_hbm_avail_gb=5.52GiB
TPU KV cache size: 321,280 tokens, Maximum concurrency for 131,072 tokens per request: 2.45x
init engine (profile, create kv cache, warmup model) took 503.54 s (compilation: 413.41 s)
Application startup complete.
```

| `--max-model-len` | 8,192 | 131,072 |
|---|---:|---:|
| KV cache, tokens | 321,280 | 321,280 |
| Full-length requests at once | 39.22 | 2.45 |
| Engine start | 502.84 s | 503.54 s |
| of which XLA compile | 408.59 s | 413.41 s |

One full-length request needs 1.875 GiB of KV cache. `scripts/longctx.py` in `tpu-survey` buries a passphrase at a chosen depth in up to 124,949 tokens of filler, starts each prompt with a unique line so vLLM's prefix cache cannot reuse an earlier run, and asks for the passphrase:

```text
$ python3 scripts/longctx.py $TPU_URL 8000 0.5
{"prompt_tokens": 7949, "completion_tokens": 10, "depth": 0.5, "seconds": 0.61, "prompt_tokens_per_second": 13110.4, "answer": "violet-harbor-7341", "found": true}
$ python3 scripts/longctx.py $TPU_URL 120000 0.1
{"prompt_tokens": 124949, "completion_tokens": 10, "depth": 0.1, "seconds": 9.53, "prompt_tokens_per_second": 13116.9, "answer": "violet-harbor-7341", "found": true}
$ python3 scripts/longctx.py $TPU_URL 120000 0.5
{"prompt_tokens": 124949, "completion_tokens": 10, "depth": 0.5, "seconds": 9.54, "prompt_tokens_per_second": 13102.3, "answer": "violet-harbor-7341", "found": true}
$ python3 scripts/longctx.py $TPU_URL 120000 0.9
{"prompt_tokens": 124949, "completion_tokens": 10, "depth": 0.9, "seconds": 9.41, "prompt_tokens_per_second": 13274.6, "answer": "violet-harbor-7341", "found": true}
$ python3 scripts/longctx.py $TPU_URL 140000 0.5
HTTP 400: {"error":{"message":"This model's maximum context length is 131072 tokens. However, you requested 16 output tokens and your prompt contains at least 131057 input tokens, for a total of at least 131073 tokens. ..."}}
```

The passphrase comes back at 10%, 50% and 90% depth of a 124,949-token prompt, in 9.4 to 9.5 seconds each, about 13,100 prompt tokens a second as timed from the client. vLLM's own log reads 12,495 tokens a second, a 0.0% prefix cache hit rate, and 38 to 39% of the KV cache in use while the request runs. A prompt past the limit is refused before it reaches the chip.

---

#### 🔎 Tip: Raise the Context on the Chip You Have

Flex-start capacity is granted per request. Replacing the container keeps the node, and its 4-hour clock, and costs one engine start, 503.54 s on this v5e. Deleting the node to relaunch it means a new flex-start request, and a new wait for capacity.

---

#### Step 4 — Precision: What the Chip Computes

A smaller format saves memory and bandwidth on every chip. It saves compute only where the matrix unit multiplies that format natively.

| Format | v5e | v6e | v5p | TPU7x |
|---|---|---|---|---|
| bf16 | native | native | native | native |
| int8 | 2x bf16 | 2x bf16 | native | native |
| fp8 | storage | storage | bf16 rate | 2x bf16 |
| int4 | storage | storage | storage | storage |

"Storage" means the weights widen to bf16 before every matrix multiply. On v5e that ordering shows up directly in E2B throughput at one stream:

| E2B format on v5e-1 | tok/s |
|---|---:|
| bf16 | 144 |
| 4-bit QAT + 4-bit embeddings | 150 |
| fp8 | 182 |
| fp8 + 4-bit embeddings | 203 |
| int8 W8A8 | 220 |
| int8 + 4-bit embeddings | 243 |

Accuracy holds: 0.683 for bf16 against 0.677 for int8 with 4-bit embeddings, on a 3,880-record suite. At 12B, fp8 serves 518 tok/s at 16 streams against 624 for int8 at the same file size.

On v6e the 4-bit path pays off through a fused kernel. Gemma 4 26B-A4B in 4-bit QAT W4A16 serves 1,283 tok/s at 16 streams against 668 for fp8, in 17.43 GiB of weights against 27.99 GiB, and holds 53,888 KV-cache tokens against 3,456. Accuracy is 75.3% against 76.0%. The stock image refuses W4A16; it took tpu-inference PRs #3299, #3653 and #3660.

---

#### Step 5 — Memory and Bandwidth

Each generated token reads every weight once, so bandwidth sets the ceiling for one request.

| Chip | Memory | Bandwidth | E2B bf16 tok/s, 1 stream | tok/s per TB/s |
|---|---:|---:|---:|---:|
| TPU v5e | 16 GB | 859 GB/s | 120 | 140 |
| TPU v6e | 32 GB | 1,638 GB/s | 199 | 122 |
| TPU v5p | 95 GiB | 2,765 GB/s | not measured | — |
| TPU7x | 192 GiB | 7,380 GB/s | not measured | — |
| NVIDIA L4 | 24 GB | 300 GB/s | 46 | 154 |

The three measured chips land between 122 and 154 tokens a second per TB/s. Google quotes v5e bandwidth as 800 GiBps, which is 859 GB/s. The L4 figure was measured on an AWS g6.2xlarge, the same chip as Google Cloud's g2.

---

#### Step 6 — Topologies and Slices

Weight size picks the slice. Every Gemma 4 size serves on one host:

| Model | v5e-1 | v6e-1 | v6e-4, TP=4 |
|---|---|---|---|
| E2B | bf16 or int8 | bf16 | — |
| E4B | int8 or 4-bit; bf16 does not fit | bf16 | — |
| 12B | int8 or 4-bit | bf16, 22.18 GiB | — |
| 26B-A4B | 4-bit, 201 tok/s | 4-bit, 1,283 tok/s | bf16, 1,982 tok/s |
| 31B | does not fit | 4-bit, 500 tok/s | bf16, 2,031 tok/s |

Small models scale out. On a v5e-4, four one-chip E2B engines serve 1.39x the output of one TP=4 engine at 16 streams, 2.52x at 64 and 3.36x at 256. E2B has one KV head, so tensor parallelism keeps four copies of the KV cache. 12B gains from TP=4: 2.25x the v5e-1 output at one stream, and 122,624 KV tokens against 9,728.

---

#### 🔎 Tip: Quota Does Not Mean Availability

A TPU create passes three checks: hardware in the zone, the provisioning model offered there, and capacity at that moment. Quota answers none of them. 37 zones report v6e quota; 18 offer v6e-1 hardware. Three of five zones holding 1,536 chips of quota returned stockout. us-central1-b has hardware, quota and a flex-start price, and rejects flex-start. Check capacity with a spot request: it fails fast with "reason: stockout", where a flex-start request with no quota queues as PENDING exactly like a stockout.

Ask for fewer chips. In europe-west4-a a v6e-8 flex-start request expired after 26 minutes, and a v6e-4 was granted at once less than an hour later.

---

#### Step 7 — Against Other Accelerators

On the same 26B 4-bit checkpoint, with the same flags and client:

| | NVIDIA L4 | TPU v6e-1 |
|---|---:|---:|
| Output tok/s, 16 x 256 tokens | 393.7 | 1,283.3 |
| KV cache tokens | 17,990 | 53,888 |
| 3,880-record suite, wall time | 139 s | 58 s |
| 4-bit kernel | Marlin | gmm_v2 |
| Software | stock vLLM 0.30.0 | three tpu-inference patches |
| List price per hour | $0.71 (g2-standard-4) | $2.70 |

Cloud Run serves the L4 as a managed service: one L4 per instance, weights mounted from Cloud Storage at /mnt/models, a startup check that waits 180 s before its first check, and a default that scales between zero and one instance, so an idle service means a cold GPU start.

Blackwell on Google Cloud lists above v6e per accelerator-hour: an RTX PRO 6000 on g4-standard-48 at $4.50 on-demand and $2.25 flex-start, and a B200 at $8.06 per GPU flex-start ($64.44 for the 8-GPU a4-highgpu-8g, divided by 8). No rig has measured either.

Per dollar, on E2B bf16 with `vllm bench serve`, v5e at a $0.58 spot read leads to 16 streams and the MI300X at $1.99 leads at 64:

| tok/s per $/hr | MI300X ($1.99) | v5e-1 spot ($0.58) | v6e-1 ($2.97) |
|---|---:|---:|---:|
| 1 stream | 153 | 208 | 67 |
| 4 streams | 487 | 711 | 233 |
| 16 streams | 1,333 | 1,572 | 587 |
| 64 streams | 2,713 | 1,972 | 719 |

Each software stack fails in its own way:

| | TPU, vLLM + JAX | CUDA, L4 | ROCm, MI300X | Neuron, inf2 / trn1 |
|---|---|---|---|---|
| Start to healthy, E2B | 510 s v6e, 986 s v5e | 334 s | about 160 s | 171 s on trn1 |
| Gemma 4 support | 4-bit needed patches | 26B 4-bit loads stock | fp8 only as fnuz | vLLM-Neuron cannot load it |
| Hazard | OOM after a long compile | sm_89 missing; runs sm_86 code | AITER flag 0.97 to 0.99x | large gather returns zeros, HTTP 200 |

---

#### 🔎 Tip: Cache the Compile

XLA compile is 81 to 91% of vLLM's engine start for E2B: 329 of 404 seconds on v6e-1 and 738 of 814 on v5e-1, inside 510 and 986 seconds from container start to healthy. On today's v5e-1 with `vllm/vllm-tpu:v0.30.0` the engine start was 503.54 seconds, 413.41 of them compile. A mounted compile cache cut a v5e restart from 857 to 497 seconds, 42%. 31B weights load in 5.6 minutes from in-region Cloud Storage against 60 to 90 from Hugging Face.

Serving flags that work on GPUs fail late on TPU, after the compile. `--gpu-memory-utilization 0.95` failed on v5e after a 691-second compile, twice; 0.92 serves. `--kv-cache-dtype fp8` logs fp8 on v6e and allocates bf16. The `vllm/vllm-tpu:nightly` tag broke two 4-bit builds on 2026-10-02, so pin an image digest.

---

#### Step 8 — What Each Chip Costs

| $ per chip-hour | On-demand | Flex-start | Calendar | 1-yr CUD | 3-yr CUD | Spot read |
|---|---:|---:|---:|---:|---:|---:|
| v5e | $1.20 | $0.60 | $0.84 | $0.84 | $0.54 | $0.58 |
| v6e Trillium | $2.70 | $1.35 | $1.89 | $1.89 | $1.22 | $1.40 |
| v5p | $4.20 | $2.10 | $2.94 | $2.94 | $1.89 | — |
| TPU7x Ironwood | $12.00 | $6.00 | $8.40 | $8.40 | $5.40 | — |

Flex-start halves on-demand on every generation and can undercut spot: v6e in us-east5 lists $1.35 flex-start against a $1.40 spot read. Flex-start runs 10 minutes to 7 days and bills until deleted.

Utilization sets the price per token. A v6e-1 at $2.97 an hour serves E2B at $4.14 per million output tokens with one stream, $0.70 with eight and $0.55 with 32. A v5e at flex-start goes from $1.31 at one stream to $0.09 at 64.

---

#### Compare and Contrast

| Option | Precision served | Bandwidth | Measured result | List $/hr | Software |
|---|---|---:|---|---:|---|
| 🥇 TPU v6e-1 | 4-bit QAT W4A16 | 1,638 GB/s | 26B at 1,283 tok/s | $2.70 | tpu-inference, patched |
| 🥈 TPU v5e-1 | int8 W8A8 + 4-bit embeddings | 859 GB/s | E2B at 243 tok/s, one stream | $1.20 | tpu-inference |
| 🥉 NVIDIA L4 | 4-bit QAT | 300 GB/s | 26B at 394 tok/s | $0.71 | stock vLLM |
| TPU7x Ironwood | fp8 native | 7,380 GB/s | not measured | $12.00 | allowlisted |
| Blackwell B200 | fp4 native | — | not measured | $8.06 flex, per GPU | not measured |

---

#### So, Which One?

For E2B or E4B at the lowest cost, run v5e-1 in int8 with one engine per chip, on spot or flex-start. For 12B to 31B on one chip, run v6e-1 in 4-bit QAT. For bf16 26B or 31B with long context, run v6e-4 with TP=4. For stock software and the fastest boot, run the L4, on Compute Engine or on Cloud Run. TPU7x brings fp8 math and 192 GiB per chip, behind an allowlist. Whichever you pick: check capacity with a spot request, ask for the smallest slice that fits, pin the image digest and mount the compile cache.

---

#### Summary

The goal of this article was to serve Gemma 4 on a Cloud TPU in one command, then explain the results across TPU generations through native precision, memory bandwidth, slice shape and capacity. The key to the solution was measuring each claim on the rigs, with every ratio and price computed in code. The results were:

- 🟢 **One v5e chip holds Gemma 4 E2B's full 131,072-token context**: a 321,280-token KV cache beside 8.97 GiB of bf16 weights, and a passphrase found at 10%, 50% and 90% depth of a 124,949-token prompt in 9.4 to 9.5 s.
- 🟢 **The agent sends the right filter**: 3 of 3 runs on the v5e called `count_ids` with `id >= 10` and quoted 8.
- 🟢 **int8 is the fast format on v5e**: with 4-bit embedding tables it serves E2B at 1.7x bf16, with accuracy within a point.
- 🟢 **v6e-1 serves 3.3x the L4's tokens** on the same 26B 4-bit checkpoint, with 3x the KV cache.
- 🟢 **Flex-start halves on-demand** on every generation, and can undercut spot.
- ⚠️ **Cold start takes 8.5 to 16.4 minutes**, most of it XLA compile; cache it.
- ⚠️ **One-stream decode follows bandwidth**: 122 to 154 tok/s per TB/s on v5e, v6e and the L4.
- ❌ **Quota does not reserve hardware**: 37 zones report v6e quota, 18 have v6e-1 hardware, and full-quota zones still stock out.

Scope: the hands-on steps ran on one flex-start v5e-1 in us-west4-a on 2026-10-25, `google/gemma-4-E2B-it` in bf16 on `vllm/vllm-tpu:v0.30.0`, three demo runs and one long-context request per depth. The rest covers Gemma 4 E2B to 31B on Cloud TPU v5e and v6e, measured in the gemma4-dev rigs between July and October 2026 with vLLM and tpu-inference; L4 on Google Cloud g2 and AWS g6; MI300X on DigitalOcean; Inferentia2 and Trainium on AWS. The cross-vendor tokens-per-dollar table mixes vLLM builds, hosts and repeat counts, so it shows direction. v5p, TPU7x and Blackwell figures are Google's published specifications and list prices, read on 2026-10-12 and re-read on 2026-10-25 with the same values; spot prices are dated Billing Catalog reads.

The strategy for serving Gemma 4 on Google Cloud from v5e to Ironwood was validated with an incremental step by step approach.

---

#### References

* [gemma4-dev | GitHub](https://github.com/xbill9/gemma4-dev)
* [gcp-inference | GitHub](https://github.com/xbill9/gcp-inference)
* [tpu-survey, long-context test and evidence | GitHub](https://github.com/xbill9/tpu-survey)
* [Gemma 4 E2B-it | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it)
* [Inference on Google Cloud slides | Google Slides](https://docs.google.com/presentation/d/1hT8UXOEBueZfoFdIH0ZyhYhQ3WZJ_0xbCsvNDDs8H4A/edit)
* [Inference on AWS slides | Google Slides](https://docs.google.com/presentation/d/1Ax8KGF8T0EYffshY7WfUM61QcHmLk0q72EE7G_hIwgM/edit)
* [tpu-inference | GitHub](https://github.com/vllm-project/tpu-inference)
* [Cloud TPU documentation | Google Cloud](https://docs.cloud.google.com/tpu/docs)
* [TPU7x (Ironwood) | Google Cloud](https://docs.cloud.google.com/tpu/docs/tpu7x)
* [Cloud TPU pricing | Google Cloud](https://cloud.google.com/tpu/pricing)
* [Accelerator-optimized machine pricing | Google Cloud](https://cloud.google.com/products/compute/pricing/accelerator-optimized)
* [In-Datacenter Performance Analysis of a Tensor Processing Unit | arXiv](https://arxiv.org/abs/1704.04760)
