# Figure trace: devto-gcp-inference-companion.md

Every figure in `/home/xbill/tpu-survey/devto-gcp-inference-companion.md`, from "What Is the Problem?" through Summary and Scope, traced to a text artifact. Traced 2026-10-25.

Source paths are relative to `/home/xbill/`. `gemma4-dev/QUANTIZATION.md`, `HARDWARE.md` and `TPU.md` are byte-identical to their copies in `gemma4-dev/docs/inference-engineering/evidence/` (checked with `diff -q`), so the root-file line numbers hold for both. The excerpt for each source is in this directory, named after the path with `/` replaced by `__` and `.txt` appended. Arithmetic is recomputed in `arithmetic.py`; its output is `arithmetic.txt`.

Verdicts: **MATCH**: the source line states the figure, or the figure follows from the cited inputs by the arithmetic shown. **DIFFERS**: what the source says instead. **NOT FOUND**: no .md/.json/.txt/.csv source in `gemma4-dev` (or the `gcp-inference` price archive) states it.

| # | Figure | Article section | Source path:line | Verdict |
|---:|---|---|---|---|
| 1 | 65 Cloud TPU rigs | Intro | none | NOT FOUND. No file states 65. `ls` of `gemma4-dev` finds 66 directories named `tpu-*`, `gce-*` or `gke-*` on TPU hardware (inf2 and trn1 excluded). Two of them are v5p rigs, and HARDWARE.md:221 says nothing was booted on v5p. Which of the 66 measured something was not checked. |
| 2 | Measured between August and October 2026 | Intro, Scope | gemma4-dev/tpu-jax-v6e1-2b/benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json:197-199 | DIFFERS: the v6e-1 cold-start figures (510 s, 329 s compile) come from a report dated 2026-07-21. |
| 3 | Four TPU generations | Problem | none | NOT FOUND as a stated count. The article's own table lists v5e, v6e, v5p and TPU7x. |
| 4 | Five ways to pay | Problem | gemma4-dev/docs/inference-engineering/evidence/gcp-OPTIONS.md:25 | NOT FOUND for TPU. The only "five ways to pay" line is about Gemini on Agent Platform, which is a different product. The article's Step 5 table has six price columns. |
| 5 | E2B 144 tok/s, v5e, bf16 | Problem | gemma4-dev/QUANTIZATION.md:608 | MATCH (`144 / 560 / 2,008` at 1 / 4 / 16 requests) |
| 6 | E2B 243 tok/s, int8 + 4-bit embeddings | Problem | gemma4-dev/QUANTIZATION.md:614 | MATCH (`w8a8emb4`, 243 at 1 request) |
| 7 | v6e-1 3.3x the L4's tokens, 26B | Problem | gemma4-dev/jev-tpu-31b/devto-gemma4-26b-qat-w4a16-v6e1.md:144,188 | MATCH: 1283.3/393.7 = 3.26 |
| 8 | Quota for 1,536 chips, still stockout | Problem | gemma4-dev/gce-vllm-v6e8-31b/CLAUDE.md:206-214 | MATCH |
| 9 | v5e 16 GB HBM | Generations | gemma4-dev/HARDWARE.md:42 | MATCH |
| 10 | v5e 197 TFLOPs bf16 | Generations | gemma4-dev/HARDWARE.md:44 | MATCH |
| 11 | v5e int8 393 TOPs | Generations | gemma4-dev/HARDWARE.md:45 | MATCH |
| 12 | v5e max pod 256 | Generations | none | NOT FOUND |
| 13 | v5e $1.20/hr on-demand | Generations | gemma4-dev/HARDWARE.md:51; gemma4-dev/tpu-vllm-v5e1-2b/v5e.md:291 | MATCH (`$1.2000`, us-west4) |
| 14 | v6e 32 GB HBM | Generations | gemma4-dev/HARDWARE.md:42 | MATCH |
| 15 | v6e 918 TFLOPs bf16 | Generations | gemma4-dev/HARDWARE.md:44 | MATCH |
| 16 | v6e int8 1,836 TOPs | Generations | gemma4-dev/HARDWARE.md:45 | MATCH |
| 17 | v6e max pod 256 | Generations | gemma4-dev/HARDWARE.md:50 | MATCH (`234.9 PFLOPs (256 chips)`) |
| 18 | v6e $2.70/hr on-demand | Generations | gemma4-dev/HARDWARE.md:51,73-74 | MATCH (US regions) |
| 19 | v5p 95 GiB | Generations, Step 2 | gemma4-dev/HARDWARE.md:225 | MATCH |
| 20 | v5p 459 TFLOPs bf16 | Generations | gemma4-dev/HARDWARE.md:227 | MATCH |
| 21 | v5p fp8 at the bf16 rate | Generations, Step 1 | gemma4-dev/HARDWARE.md:16 | MATCH ("bf16 and FP8 at the same 459 TFLOPs") |
| 22 | v5p max pod 8,960 | Generations | gemma4-dev/HARDWARE.md:233 | MATCH (the same row also says 6,144 max schedulable) |
| 23 | v5p $4.20/hr on-demand | Generations, Step 5 | none | NOT FOUND |
| 24 | TPU7x 192 GiB | Generations, Step 2, So Which One | gemma4-dev/docs/inference-engineering/evidence/gcp-OPTIONS.md:82 | DIFFERS: source says "192 GB" |
| 25 | TPU7x 2,307 TFLOPs bf16 | Generations | none | NOT FOUND |
| 26 | TPU7x fp8 4,614 TFLOPs | Generations | gemma4-dev/docs/inference-engineering/evidence/gcp-OPTIONS.md:82 | NOT FOUND. The source gives only "fp8 at twice bf16". 4,614 = 2 x 2,307, and 2,307 is itself unsourced (row 25). |
| 27 | TPU7x max pod 9,216 | Generations | none | NOT FOUND |
| 28 | TPU7x $12.00/hr on-demand | Generations, Step 5, Compare | none | NOT FOUND |
| 29 | First TPU in 2015, for 8-bit inference | Generations | none | NOT FOUND in the repo. The article cites arXiv 1704.04760, which is not archived here. |
| 30 | Ironwood launched in 2025, "first TPU for the age of inference" | Generations | none | NOT FOUND |
| 31 | int8 = 2x bf16 on v5e and v6e | Step 1 | gemma4-dev/HARDWARE.md:15,17,22-23,45 | MATCH |
| 32 | v5p int8 native | Step 1 | gemma4-dev/HARDWARE.md:16 | MATCH (int8 "yes") |
| 33 | TPU7x fp8 = 2x bf16 | Step 1 | gemma4-dev/docs/inference-engineering/evidence/gcp-OPTIONS.md:82; gemma4-dev/HARDWARE.md:18 | MATCH |
| 34 | fp8 and int4 are storage only on v5e and v6e; int4 is storage only on v5p and TPU7x | Step 1 | gemma4-dev/HARDWARE.md:27-30 | MATCH for v5e/v6e. NOT FOUND for int4 on v5p and TPU7x. |
| 35 | E2B v5e-1 bf16 144 tok/s | Step 1 | gemma4-dev/QUANTIZATION.md:608 | MATCH |
| 36 | E2B v5e-1 "4-bit QAT" 150 tok/s | Step 1 | gemma4-dev/QUANTIZATION.md:611 | DIFFERS: 150 is `q4w4a16emb4`, which is 4-bit QAT **with int4 embedding tables**. The other 4-bit builds score 144 (`q4_0`, stored bf16, :609) and 137 (`q4w4a16ple4`, :610). |
| 37 | E2B v5e-1 fp8 182 tok/s | Step 1 | gemma4-dev/QUANTIZATION.md:615 | MATCH |
| 38 | E2B v5e-1 fp8 + 4-bit embeddings 203 tok/s | Step 1 | gemma4-dev/QUANTIZATION.md:616 | MATCH |
| 39 | E2B v5e-1 int8 W8A8 220 tok/s | Step 1 | gemma4-dev/QUANTIZATION.md:612 | MATCH |
| 40 | E2B v5e-1 int8 + 4-bit embeddings 243 tok/s | Step 1 | gemma4-dev/QUANTIZATION.md:614 | MATCH |
| 41 | Accuracy 0.683 bf16 | Step 1 | gemma4-dev/QUANTIZATION.md:608 | MATCH |
| 42 | Accuracy 0.677 int8 + 4-bit embeddings | Step 1 | gemma4-dev/QUANTIZATION.md:614 | MATCH |
| 43 | 3,880-record suite | Step 1, Step 4 | gemma4-dev/tpu-vllm-v5e1-2b-w8a8emb4/README.md:10; gemma4-dev/jev-tpu-31b/devto-gemma4-26b-qat-w4a16-v6e1.md:143 | MATCH |
| 44 | 12B fp8 518 tok/s at 16 streams | Step 1 | gemma4-dev/QUANTIZATION.md:627,640 | MATCH |
| 45 | 12B int8 624 tok/s, same file size | Step 1 | gemma4-dev/QUANTIZATION.md:625,640 | MATCH (`w8a8`; "518 against 624 tok/s at 16") |
| 46 | 26B 4-bit QAT W4A16 1,283 tok/s at 16 streams | Step 1 | gemma4-dev/QUANTIZATION.md:353,355 | MATCH |
| 47 | 26B fp8 668 tok/s | Step 1 | gemma4-dev/QUANTIZATION.md:352 | MATCH |
| 48 | 17.43 GiB vs 27.99 GiB of weights | Step 1 | gemma4-dev/QUANTIZATION.md:352-353 | MATCH. The source column is "HBM used". |
| 49 | 53,888 vs 3,456 KV tokens | Step 1 | gemma4-dev/QUANTIZATION.md:352-353 | MATCH |
| 50 | Accuracy 75.3% vs 76.0% | Step 1 | gemma4-dev/QUANTIZATION.md:352-353; gemma4-dev/jev-tpu-31b/results/2026-09-26-26b-ARTICLE-FIGURES.md:26 | MATCH (W4A16 vs FP8) |
| 51 | tpu-inference PRs #3299, #3653, #3660 | Step 1 | gemma4-dev/QUANTIZATION.md:210-213 | MATCH |
| 52 | v5e 16 GB, 859 GB/s | Step 2 | gemma4-dev/HARDWARE.md:42-43,82 | MATCH (800 GiBps = 858.99 GB/s) |
| 53 | v6e 32 GB, 1,638 GB/s | Step 2 | gemma4-dev/HARDWARE.md:42-43 | MATCH |
| 54 | v5p 2,765 GB/s | Step 2 | gemma4-dev/HARDWARE.md:226 | MATCH |
| 55 | TPU7x 7,380 GB/s | Step 2, Compare | gemma4-dev/docs/inference-engineering/evidence/gcp-OPTIONS.md:82 | MATCH (`7380 GBps`) |
| 56 | L4 24 GB, 300 GB/s | Step 2, Compare | gemma4-dev/docs/inference-engineering/evidence/gcp-OPTIONS.md:73 | MATCH |
| 57 | v5e E2B bf16, one stream, 120 tok/s | Step 2 | gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:21 | MATCH (`c1=120.0`, in=1024 out=128) |
| 58 | v6e E2B bf16, one stream, 199 tok/s | Step 2 | gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:22 | MATCH (`c1=199.4`) |
| 59 | L4 E2B bf16, one stream, 46 tok/s | Step 2 | gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:23 | MATCH (`c1=46.1`) |
| 60 | v5e 140 tok/s per TB/s | Step 2 | arithmetic.txt | MATCH: 120.0/0.859 = 139.7 |
| 61 | v6e 121 tok/s per TB/s | Step 2 | arithmetic.txt | DIFFERS: 199.4/1.638 = 121.7, which rounds to 122. With the rounded 199 it is 121.5. |
| 62 | L4 153 tok/s per TB/s | Step 2 | arithmetic.txt | DIFFERS: 46.1/0.300 = 153.7, which rounds to 154. With the rounded 46 it is 153.3. |
| 63 | "between 121 and 153" per TB/s | Step 2, Summary | arithmetic.txt | DIFFERS: the inputs give 121.7 to 153.7, which round to 122 and 154. |
| 64 | Google quotes 800 GiBps = 859 GB/s | Step 2 | gemma4-dev/HARDWARE.md:80-82 | MATCH |
| 65 | L4 figure from AWS g6.2xlarge | Step 2 | gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:11 | MATCH |
| 66 | E2B/E4B on v5e-1: bf16 or int8 | Step 3 | gemma4-dev/HARDWARE.md:104-105; gemma4-dev/QUANTIZATION.md:600-601 | DIFFERS for E4B: "only E2B fits at bf16" on v5e-1, and E4B bf16 "do[es] not fit v5e". |
| 67 | 12B on v5e-1: int8 or 4-bit | Step 3 | gemma4-dev/QUANTIZATION.md:623,625,644 | MATCH |
| 68 | 12B on v6e-1: bf16, 22.18 GiB | Step 3 | gemma4-dev/QUANTIZATION.md:301 | MATCH |
| 69 | 26B on v5e-1: 4-bit, 201 tok/s | Step 3 | gemma4-dev/tpu-vllm-v5e1-26b-q4w4a16/README.md:19 | MATCH (`31 / 102 / 201` at 1 / 4 / 16) |
| 70 | 26B on v6e-1: 4-bit, 1,283 tok/s | Step 3 | gemma4-dev/QUANTIZATION.md:353 | MATCH |
| 71 | 26B on v6e-4 TP=4: bf16, 1,982 tok/s | Step 3 | gemma4-dev/QUANTIZATION.md:360-363; gemma4-dev/jev-tpu-31b/results/2026-09-26-26b-ARTICLE-FIGURES.md:94 | MATCH |
| 72 | 31B does not fit v5e-1 | Step 3 | gemma4-dev/QUANTIZATION.md:240; gemma4-dev/HARDWARE.md:102 | MATCH by arithmetic: 4-bit 31B takes 21.67 GiB, more than the 14.49 GiB usable. No source states it outright. |
| 73 | 31B on v6e-1: 4-bit, 500 tok/s | Step 3 | gemma4-dev/QUANTIZATION.md:240 | MATCH |
| 74 | 31B on v6e-4 TP=4: bf16, 2,031 tok/s | Step 3 | gemma4-dev/gce-vllm-v6e8-31b/benchmarks/runs/2026-08-25-tp4-v6e4-31b-validation/REPORT.md:11-15,69-73 | MATCH. Load: batch 256, 32-token prompts, 128 output, on the `:nightly` image. That differs from the 16 x 256 load in the other cells. |
| 75 | v5e-4: four engines 1.39x TP=4 at 16 streams | Step 3 | gemma4-dev/docs/inference-engineering/evidence/tpu-tp-vs-replicas-v5e4.md:80 | MATCH (2033.2/1464.6 = 1.388). The model is the E2B W8A8 int8 checkpoint. |
| 76 | 2.52x at 64 | Step 3 | gemma4-dev/docs/inference-engineering/evidence/tpu-tp-vs-replicas-v5e4.md:81 | MATCH (5471.8/2167.5 = 2.524) |
| 77 | 3.36x at 256 | Step 3 | gemma4-dev/docs/inference-engineering/evidence/tpu-tp-vs-replicas-v5e4.md:82 | MATCH (8158.5/2426.2 = 3.363) |
| 78 | E2B has one KV head | Step 3 | gemma4-dev/docs/inference-engineering/evidence/tpu-tp-vs-replicas-v5e4.md:24 | MATCH |
| 79 | 12B TP=4: 2.25x v5e-1 at one stream | Step 3 | gemma4-dev/tpu-vllm-v5e4-12b-w8a8emb4/README.md:33 | MATCH (128.4/57.0 = 2.25) |
| 80 | 122,624 vs 9,728 KV tokens | Step 3 | gemma4-dev/tpu-vllm-v5e4-12b-w8a8emb4/README.md:32 | MATCH |
| 81 | A create passes three checks | Tip: Quota | gemma4-dev/tpu-vllm-v6e1-2b/tpu_zones_status.md:20-50 | MATCH ("Three separate gates") |
| 82 | 37 zones report v6e quota | Tip: Quota, Summary | gemma4-dev/tpu-vllm-v6e1-2b/tpu_zones_status.md:24 | MATCH |
| 83 | 18 offer v6e-1 hardware | Tip: Quota, Summary | gemma4-dev/tpu-vllm-v6e1-2b/tpu_zones_status.md:24 | MATCH |
| 84 | Three of five 1,536-chip zones returned stockout | Tip: Quota | gemma4-dev/gce-vllm-v6e8-31b/CLAUDE.md:206-214 | MATCH (us-central1-a, us-central1-c and us-west1-c returned `reason: stockout`; us-central1-b provisioned, then stocked out) |
| 85 | us-central1-b has hardware, quota and a flex-start price, and rejects flex-start | Tip: Quota | gemma4-dev/tpu-vllm-v6e1-2b/tpu_zones_status.md:41-44 | MATCH |
| 86 | Spot fails fast with "reason: stockout"; flex-start with no quota sits PENDING like a stockout | Tip: Quota | gemma4-dev/TPU.md:221-227; gemma4-dev/gce-vllm-v6e8-31b/CLAUDE.md:219 | MATCH |
| 87 | v6e-8 flex-start in europe-west4-a expired after 26 minutes | Tip: Quota | gemma4-dev/gce-vllm-v6e8-31b/benchmarks/runs/2026-08-25-tp4-v6e4-31b-validation/REPORT.md:113 | MATCH |
| 88 | v6e-4 granted at once, under an hour later | Tip: Quota | gemma4-dev/gce-vllm-v6e8-31b/benchmarks/runs/2026-08-25-tp4-v6e4-31b-validation/REPORT.md:116-118 | MATCH |
| 89 | L4 393.7 tok/s, 16 x 256 | Step 4 | gemma4-dev/jev-tpu-31b/devto-gemma4-26b-qat-w4a16-v6e1.md:188; gemma4-dev/QUANTIZATION.md:389-390 | MATCH |
| 90 | v6e-1 1,283.3 tok/s, 16 x 256 | Step 4 | gemma4-dev/jev-tpu-31b/devto-gemma4-26b-qat-w4a16-v6e1.md:144; gemma4-dev/jev-tpu-31b/results/2026-09-26-26b-ARTICLE-FIGURES.md:97 | MATCH |
| 91 | L4 KV 17,990 tokens | Step 4 | gemma4-dev/QUANTIZATION.md:389; gemma4-dev/jev-tpu-31b/devto-gemma4-26b-qat-w4a16-v6e1.md:187 | MATCH |
| 92 | v6e-1 KV 53,888 tokens | Step 4 | gemma4-dev/QUANTIZATION.md:353 | MATCH |
| 93 | L4 suite wall time 139 s | Step 4 | none of the allowed types | NOT FOUND in .md/.json/.txt/.csv. The figure appears only in `gemma4-dev/jev-tpu-31b/results/2026-09-26-gpu-l4-evidence/run.log:166` ("suite: 139s for 3880 records"), a .log file. |
| 94 | v6e-1 suite wall time 58 s | Step 4 | gemma4-dev/jev-tpu-31b/devto-gemma4-26b-qat-w4a16-v6e1.md:143 | MATCH ("26b-q4w4 suite: 58s for 3880 records") |
| 95 | Marlin vs gmm_v2 | Step 4 | gemma4-dev/QUANTIZATION.md:388-389,393-394 | MATCH |
| 96 | Stock vLLM 0.30.0 on the L4 | Step 4 | gemma4-dev/QUANTIZATION.md:386 | MATCH |
| 97 | Three tpu-inference patches | Step 4 | gemma4-dev/QUANTIZATION.md:210-213 | MATCH (#3299, #3653, #3660) |
| 98 | L4 $0.71/hr (g2-standard-4) | Step 4, Compare | gcp-inference/results/2026-10-12/prices.json:76-92 | MATCH by arithmetic: 4 x 0.024988 + 16 x 0.002927 + 0.560040 = 0.7068 (us-central1, read 2026-10-12). The archive is in `gcp-inference`, not `gemma4-dev`. The 26B L4 run itself used g2-standard-8 (QUANTIZATION.md:386). |
| 99 | v6e $2.70/hr | Step 4, Compare | gemma4-dev/HARDWARE.md:51 | MATCH |
| 100 | Cloud Run: one L4 per instance | Step 4 | gemma4-dev/docs/inference-engineering/evidence/gcp-OPTIONS.md:53 | MATCH |
| 101 | Weights mounted at /mnt/models | Step 4 | gemma4-dev/gpu-2B-cloudrun-devops-agent/docs/evidence/make-deploy.txt:13-14 | MATCH |
| 102 | Startup probe waits 180 s | Step 4 | gemma4-dev/gpu-2B-cloudrun-devops-agent/docs/evidence/make-deploy.txt:5 | MATCH (`initialDelaySeconds=180`) |
| 103 | Scales between zero and one instance | Step 4 | gemma4-dev/gpu-2B-cloudrun-devops-agent/docs/evidence/make-deploy.txt:7-8 | MATCH |
| 104 | RTX PRO 6000 on g4-standard-48 at $4.50 on-demand | Step 4 | gemma4-dev/docs/inference-engineering/evidence/gcp-OPTIONS.md:44 (machine name only) | NOT FOUND (price). Only the machine-to-GPU pairing is sourced. |
| 105 | RTX PRO 6000 $2.25 flex-start | Step 4 | none | NOT FOUND |
| 106 | B200 $8.06 per GPU flex-start | Step 4, Compare | none | NOT FOUND |
| 107 | v5e spot $0.58 | Step 4, Step 5 | gemma4-dev/tpu-vllm-v5e1-2b/v5e.md:293; gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:9 | MATCH (`$0.5779`) |
| 108 | MI300X $1.99/hr | Step 4 | gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:8; gemma4-dev/gpu-vllm-mi300x-2b/evidence/2026-09-17-deploy-and-validate.txt:11 | MATCH |
| 109 | v6e-1 $2.97/hr | Step 4, Step 5 | gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:10; gemma4-dev/HARDWARE.md:74 | MATCH (europe-west4 on-demand) |
| 110 | MI300X tok/s per $/hr: 153 / 487 / 1,333 / 2,713 | Step 4 | gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:8 | MATCH (recomputed from :20 /1.99) |
| 111 | v5e-1 spot: 208 / 711 / 1,572 / 1,972 | Step 4 | gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:9 | MATCH (recomputed from :21 /0.5779) |
| 112 | v6e-1: 67 / 233 / 587 / 719 | Step 4 | gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:10 | MATCH (recomputed from :22 /2.97) |
| 113 | v5e leads to 16 streams, MI300X leads at 64 | Step 4 | gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:8-9 | MATCH (1,572 > 1,333 at 16; 2,713 > 1,972 at 64) |
| 114 | Start to healthy, E2B: 510 s v6e | Step 4 | gemma4-dev/tpu-jax-v6e1-2b/benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json:197 | MATCH |
| 115 | Start to healthy, E2B: 986 s v5e | Step 4 | gemma4-dev/tpu-vllm-v5e1-2b/benchmarks/runs/2026-08-06-vllm-sweep-v5e1/REPORT.md:25 | MATCH |
| 116 | Start to healthy, E2B: 334 s L4 | Step 4 | gemma4-dev/gpu-vllm-g6-2b/benchmarks/runs/2026-08-30-first-serve-g6/REPORT.md:121 | MATCH (AWS g6, "bootstrap to /health 200") |
| 117 | Start to healthy, E2B: about 160 s MI300X | Step 4 | gemma4-dev/gpu-vllm-mi300x-2b/evidence/2026-09-17-deploy-and-validate.txt:29 | MATCH ("READY after 160s") |
| 118 | Start to healthy, E2B: 171 s trn1 | Step 4 | gemma4-dev/tpu-pytorch-trn1-2b/benchmarks/runs/2026-10-05-smoke-trn1/REPORT.md:23 | MATCH ("Pull to healthy", after a 365 s image pull, :22) |
| 119 | sm_89 missing; runs sm_86 code | Step 4 | gemma4-dev/gpu-vllm-g6-2b/benchmarks/runs/2026-08-30-first-serve-g6/REPORT.md:21-23 | MATCH |
| 120 | AITER flag 0.97 to 0.99x | Step 4 | gemma4-dev/docs/inference-engineering/evidence/amd-aiter-derived-tables.md:13 | MATCH (min 0.974, max 0.994) |
| 121 | fp8 only as fnuz | Step 4 | gemma4-dev/HARDWARE.md:721-722 | MATCH |
| 122 | Large gather returns zeros, HTTP 200 | Step 4 | gemma4-dev/HARDWARE.md:673-676 | MATCH |
| 123 | XLA compile is 81 to 91% of engine start | Tip: Compile | gemma4-dev/tpu-jax-v6e1-2b/benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json:197-199; gemma4-dev/tpu-vllm-v5e1-2b/benchmarks/runs/2026-08-06-vllm-sweep-v5e1/REPORT.md:25,29 | DIFFERS. 81% and 91% are compile over **engine init**: 329/404 = 81.4% and 738/814 = 90.7%. The article pairs them with time to healthy, "329 of 510" and "738 of 986", which give 64.5% and 74.8%. |
| 124 | 329 of 510 s on v6e-1 | Tip: Compile | gemma4-dev/tpu-jax-v6e1-2b/benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json:197,199 | MATCH (each number; see row 123 for the percentage) |
| 125 | 738 of 986 s on v5e-1 | Tip: Compile | gemma4-dev/tpu-vllm-v5e1-2b/benchmarks/runs/2026-08-06-vllm-sweep-v5e1/REPORT.md:25 | MATCH |
| 126 | Compile cache: 857 to 497 s, 42% | Tip: Compile | gemma4-dev/tpu-vllm-v5e1-2b/SERVING-PARAMS.md:78 | MATCH ((857-497)/857 = 42.0%) |
| 127 | 31B weights 5.6 min from in-region GCS vs 60 to 90 from HF | Tip: Compile | gemma4-dev/gce-vllm-v6e8-31b/benchmarks/runs/2026-08-25-tp4-v6e4-31b-validation/REPORT.md:103 | MATCH. The 60-90 minute figure is the pull time "the boot budget was sized for", not a measured pull. |
| 128 | 0.95 failed after a 691 s compile, twice; 0.92 serves | Tip: Compile | gemma4-dev/tpu-vllm-v5e1-2b/SERVING-PARAMS.md:42 | MATCH |
| 129 | nightly broke two 4-bit builds on 2026-10-02 | Tip: Compile | gemma4-dev/QUANTIZATION.md:542-545 | MATCH (`-qat-q4_0-unquantized` and `-qat-w4a16-ct`) |
| 130 | v5e on-demand $1.20 | Step 5 | gemma4-dev/tpu-vllm-v5e1-2b/v5e.md:291 | MATCH |
| 131 | v5e flex-start $0.60 | Step 5 | gemma4-dev/tpu-vllm-v5e1-2b/v5e.md:292 | MATCH |
| 132 | v5e calendar $0.84 | Step 5 | gemma4-dev/tpu-vllm-v5e1-2b/v5e.md:294 | MATCH |
| 133 | v5e 1-yr CUD $0.84 | Step 5 | gemma4-dev/tpu-vllm-v5e1-2b/v5e.md:295 | MATCH |
| 134 | v5e 3-yr CUD $0.54 | Step 5 | gemma4-dev/tpu-vllm-v5e1-2b/v5e.md:295 | MATCH |
| 135 | v5e spot $0.58 | Step 5 | gemma4-dev/tpu-vllm-v5e1-2b/v5e.md:293 | MATCH ($0.5779) |
| 136 | v6e on-demand $2.70 | Step 5 | gemma4-dev/HARDWARE.md:51 | MATCH |
| 137 | v6e flex-start $1.35 | Step 5 | gemma4-dev/HARDWARE.md:75 | MATCH |
| 138 | v6e calendar $1.89 | Step 5 | none | NOT FOUND. It equals 2.70 x 0.70, the v5e calendar/on-demand ratio (arithmetic.txt). |
| 139 | v6e 1-yr CUD $1.89 | Step 5 | none | NOT FOUND (also 2.70 x 0.70) |
| 140 | v6e 3-yr CUD $1.22 | Step 5 | none | NOT FOUND (2.70 x 0.45 = 1.215, the v5e ratio) |
| 141 | v6e spot $1.40 | Step 5 | gemma4-dev/HARDWARE.md:75; gemma4-dev/tpu-vllm-v6e1-2b/devto-vllm-gemma4-e2b-v6e1.md:348 | MATCH ($1.4033, us-east5) |
| 142 | v5p $4.20 / $2.10 / $2.94 / $2.94 / $1.89 | Step 5 | none | NOT FOUND (all five). They equal 4.20 x 1, 0.5, 0.7, 0.7, 0.45. |
| 143 | TPU7x $12.00 / $6.00 / $8.40 / $8.40 / $5.40 | Step 5 | none | NOT FOUND (all five). They equal 12.00 x 1, 0.5, 0.7, 0.7, 0.45. |
| 144 | Flex-start halves on-demand on every generation | Step 5, Summary | gemma4-dev/tpu-vllm-v5e1-2b/v5e.md:291-292; gemma4-dev/HARDWARE.md:51,75 | MATCH for v5e (0.60/1.20) and v6e (1.35/2.70). NOT FOUND for v5p and TPU7x (rows 142-143). |
| 145 | v6e us-east5: $1.35 flex-start vs $1.40 spot | Step 5 | gemma4-dev/HARDWARE.md:75 | MATCH |
| 146 | Flex-start runs 10 minutes to 7 days | Step 5 | gemma4-dev/docs/inference-engineering/evidence/gcp-OPTIONS.md:74 | 7 days MATCH ("up to 7 days"). 10 minutes NOT FOUND. |
| 147 | v6e-1 at $2.97: $4.14 / $0.70 / $0.55 per M at 1 / 8 / 32 streams | Step 5 | gemma4-dev/docs/inference-engineering/evidence/gke-tpu-v6e1.md:87-89,92 | MATCH (recomputed from 199 / 1180 / 1508 tok/s: 4.15 / 0.70 / 0.55) |
| 148 | v5e flex-start $1.31 at one stream, $0.09 at 64 | Step 5 | gemma4-dev/tpu-vllm-v5e1-2b/v5e.md:307,310 | MATCH ($1.314, $0.089) |
| 149 | v6e-1 row: 4-bit QAT W4A16, 1,638 GB/s, 26B at 1,283 tok/s, $2.70 | Compare | gemma4-dev/QUANTIZATION.md:353; gemma4-dev/HARDWARE.md:43,51 | MATCH |
| 150 | v5e-1 row: "int8 W8A8", E2B at 243 tok/s, one stream | Compare | gemma4-dev/QUANTIZATION.md:612,614 | DIFFERS: 243 is `w8a8emb4` (int8 W8A8 with int4 embedding tables). Plain W8A8 is 220. |
| 151 | v5e-1 row: 859 GB/s, $1.20 | Compare | gemma4-dev/HARDWARE.md:51,82 | MATCH |
| 152 | L4 row: 4-bit QAT, 26B at 394 tok/s | Compare | gemma4-dev/QUANTIZATION.md:389-390 | MATCH |
| 153 | int8 is 1.7x bf16 for E2B, within a point | Summary | gemma4-dev/QUANTIZATION.md:608,614,642 | MATCH (243/144 = 1.69; 0.683 - 0.677 = 0.6 points) |
| 154 | v6e-1 3.3x the L4's tokens, 3x the KV cache | Summary | gemma4-dev/QUANTIZATION.md:353,389 | MATCH (1283.3/393.7 = 3.26; 53,888/17,990 = 3.00) |
| 155 | Cold start 8.5 to 16.5 minutes | Summary | arithmetic.txt | DIFFERS: 510 s = 8.5 min, and 986 s = 16.4 min, not 16.5. |
| 156 | MI300X on DigitalOcean | Scope | gemma4-dev/HARDWARE.md:693; gemma4-dev/docs/inference-engineering/evidence/amd-rate-per-dollar.txt:8 | MATCH |
| 157 | Specs and list prices read 2026-10-12 | Scope | gemma4-dev/docs/inference-engineering/evidence/gcp-OPTIONS.md:3 | MATCH for the pages behind TPU7x 192 GB / 7,380 GBps. No archived copy of cloud.google.com/tpu/pricing or of a Blackwell price page exists in either repo, so the 2026-10-12 prices in rows 23, 28, 104-106 and 138-143 have no source. |

## Counts

Computed by a python3 count of the first word of each row's verdict column (157 rows):

- MATCH: 127
- DIFFERS: 10 (rows 2, 24, 36, 61, 62, 63, 66, 123, 150, 155)
- NOT FOUND: 20 (rows 1, 3, 4, 12, 23, 25, 26, 27, 28, 29, 30, 93, 104, 105, 106, 138, 139, 140, 142, 143)

Rows 34, 144 and 146 match in part and are counted as MATCH. Their NOT FOUND parts: int4 storage-only on v5p and TPU7x (34), flex-start halving on v5p and TPU7x (144), and the 10-minute minimum flex-start run (146). Rows 142 and 143 each cover five prices.
