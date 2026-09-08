# demo

Two artifacts, both run locally against Ollama (no cloud key).

## 1. OpenAI-SDK compatibility proof — `openai_sdk_check.py`

Uses the **official, unmodified** `openai` Python client against the gateway and
asserts a correct completion. If this passes, semblance is a drop-in
OpenAI-compatible endpoint.

```bash
pip install openai
# gateway running on :8088, backed by Ollama with llama3.2:1b
python demo/openai_sdk_check.py
# → PASS: unmodified OpenAI SDK works against semblance, and the answer is correct.
```

## 2. Recorded live demo — `semblance-demo.cast`

Cold miss → the policy learning (τ falling from 1.0) → a repeat served as a
**cache hit** with the headers → metrics moving.

Play it:

```bash
asciinema play demo/semblance-demo.cast
```

A rendered still is in `semblance-demo.png`. Reproduce it:

```bash
# Ollama with a chat + embedding model:
ollama pull llama3.2:1b && ollama pull nomic-embed-text

# Gateway with Ollama for both backend and embeddings, low cold-start floor:
OPENAI_API_KEY=ollama \
  SEMBLANCE_EMBED_URL=http://localhost:11434/v1 SEMBLANCE_EMBED_MODEL=nomic-embed-text \
  SEMBLANCE_EMBED_DIMENSIONS=768 SEMBLANCE_BACKEND_URL=http://localhost:11434/v1 \
  SEMBLANCE_NMIN=2 SEMBLANCE_DELTA=0.05 SEMBLANCE_LISTEN_ADDR=127.0.0.1:8088 \
  go run ./cmd/semblance &

bash demo/run_demo.sh
```

### What you're seeing, honestly

The policy is **cautious by design**: it explores while its per-entry correctness
curve is still uncertain, and τ falls as it accumulates observations. Hits begin
once it is confident enough. On this tiny hand-run that takes ~20 repeats; at
benchmark scale (tens of thousands of queries) hit rates reach the numbers in the
top-level README. `OPENAI_API_KEY=ollama` is a dummy — Ollama serves an
OpenAI-compatible `/v1/embeddings`, so no real OpenAI key is needed.
