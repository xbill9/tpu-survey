"""One long-context request to an OpenAI-compatible vLLM server, timed here.

A filler of numbered lines with one passphrase buried at a chosen depth; the model is asked
for the passphrase. Prompt size comes from the server's own usage count, not an estimate.

    python3 scripts/longctx.py http://localhost:8005/v1 120000 0.5
"""

import json
import sys
import time
import urllib.error
import urllib.request

url, target, depth = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
SECRET = "violet-harbor-7341"
# About 20 tokens per line for Gemma's tokenizer; the server's count is what gets reported.
lines = [f"Line {i}: the shipment of crates left the dock on schedule and arrived intact." for i in range(target // 20)]
lines.insert(int(len(lines) * depth), f"Line X: the passphrase is {SECRET}.")
# A fresh first line per run, so vLLM's prefix cache cannot reuse an earlier run's prompt.
prompt = f"Document {time.time_ns()}\n" + "\n".join(lines) + "\n\nWhat is the passphrase? Answer with the passphrase only."
body = json.dumps({"model": "gemma", "max_tokens": 16, "temperature": 0.0,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
req = urllib.request.Request(url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
except urllib.error.HTTPError as e:
    sys.exit(f"HTTP {e.code}: {e.read().decode()[:400]}")
secs = time.time() - t0
answer = out["choices"][0]["message"]["content"].strip()
u = out["usage"]
print(json.dumps({"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                  "depth": depth, "seconds": round(secs, 2),
                  "prompt_tokens_per_second": round(u["prompt_tokens"] / secs, 1),
                  "answer": answer, "found": SECRET in answer}))
