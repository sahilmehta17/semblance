# bench — verified-cache replay harness

Feeds the extracted SearchQueries JSONL (`{prompt, id_set, embedding}`) through
the cache decision logic in dataset order and scores the error-vs-hit-rate
tradeoff. All benchmark logic is Go; plotting is a separate presentation script
(`tools/plot_results.py`).

## What it does

- Uses the dataset's **precomputed `emb_gte` vectors directly** — no live
  embedder is called.
- Labels correctness from **ground-truth `id_set` equality** — the LLM judge is
  bypassed entirely. On a replay miss (explore), the observation label is
  `neighbor.id_set == query.id_set`.
- Scores the authors' way: on an **exploit**, TP if the matched entry's `id_set`
  equals the query's, else FP. `error rate = FP/n`, `hit rate = (TP+FP)/n`, with
  95% **Wilson** confidence intervals (the policy is randomized; the seed is
  recorded).

Two arms:

- **static** — the GPTCache/LiteLLM baseline: exploit iff `similarity >= T`, for
  `T in [0.80, 0.83, 0.86, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99]`.
- **verified** — our policy (`internal/policy`, reused verbatim): exploit iff
  `policy.Decide` says so, for `delta in [0.01, 0.02, 0.03, 0.05]`.

## Go/no-go

Every verified run's realized `FP/n` must be `<= delta`. If any exceeds it, the
harness prints the breach and **exits non-zero** — that is an implementation
bug to debug, not a result to report.

## Design notes

- The harness reuses `internal/policy` (the load-bearing statistics) but uses a
  purpose-built store with a **parallel** nearest-neighbor scan, because the
  shipped `internal/cache` scans a bucket single-threaded and this workload is
  O(n²). The store mirrors the cache's rules: normalized-cosine similarity,
  guarded insert (Algorithm 1: insert only when the neighbor was wrong), and
  observe-only-on-explore. The parallel argmax combines chunk results in index
  order, so it is bit-identical to a serial scan — runs are reproducible.
- Insertion differs per arm by design: the verified arm uses guarded insert; the
  static baseline inserts on every miss (standard semantic-cache behavior). A
  cold miss inserts unconditionally in both. This compares each method
  end-to-end, as the paper does.
- Observations per entry use a deterministic most-recent window (the shipped
  cache uses reservoir sampling; either is a valid subset and keeps the
  guarantee).

## Run

```bash
go run ./bench --input tools/prep/out/searchqueries.jsonl \
  --limit 40000 --capacity 0 --seed 42 \
  --out bench/results.json --csv bench/results.csv
```

Flags: `--limit` (queries, 0=all), `--capacity` (0=unbounded, no eviction),
`--seed`, `--workers` (default NumCPU). Nearest-neighbor is a linear scan and all
these prompts share one bucket, so cost is O(n²) — develop on ~2k, scale the
headline to taste.

Then render the plot:

```bash
tools/prep/.venv/bin/python tools/plot_results.py --results bench/results.json --out bench/error_vs_hit.png
```

## Outputs (committed)

- `results.json` — full run records (with CIs, entry counts, seed).
- `results.csv` — flat table for plotting/inspection.
- `error_vs_hit.png` — the plot, both arms with the delta budget lines drawn.
