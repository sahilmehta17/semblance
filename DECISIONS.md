# Design decisions

Why semblance is built the way it is. This records the load-bearing choices, the
places where the vCache paper is ambiguous and we picked a defensible reading,
and the systems gaps the paper leaves open that a production gateway must close.

The honest claim, stated plainly: verified semantic caching is a **published
research result** (Schroeder et al., *vCache: Verified Semantic Prompt Caching*,
ICLR 2026, arXiv:2502.03771). semblance does not invent semantic caching (that is
commodity — LiteLLM, Portkey, GPTCache) or verified thresholds (that is the
paper). What semblance contributes is a **production Go gateway on Kubernetes**
that implements the algorithm and resolves the systems problems the paper does
not address.

---

## Licensing

- The vCache **paper and source are CC BY-NC-ND**. semblance is a **clean-room**
  implementation from the paper's equations only. No vCache source file was
  read, copied, translated, adapted, or vendored — including their benchmark
  harness. Clean-room reimplementation of a published algorithm is standard and
  fine; copying their expression of it is not.
- The **datasets are Apache-2.0**, so downloading and replaying them is
  unambiguous.
- The paper is cited in the README and in the `internal/policy` package doc.

---

## Algorithm interpretations (where the paper is ambiguous)

### 1. Guarded insert — Algorithm 1 over Eq. 5

The paper's Eq. 5 inserts a new cache entry unconditionally, but **Algorithm 1
line 11 guards insertion** with `if not c(x)` — insert only when the nearest
neighbor's answer would have been *wrong*. Section 3 designates Algorithm 1 as
the exact procedure, so we follow it: a correct neighbor already covers the new
prompt, and a near-duplicate entry is wasteful. It is a config flag
(`cache.GuardedInsert`).

### 2. Objective sign in Eq. 10

The paper prints `arg min` over a log-*likelihood* in Eq. 10; taken literally
that maximizes error. It is a typo — we implement standard binary
cross-entropy **minimization** (equivalently, negative-log-likelihood).

### 3. The confidence band — penalized Fisher information, one-sided normal

The paper specifies the band only as "the CDF of the sampling distribution." We
read this as the **asymptotic-normal one-sided upper bound** on the threshold
`t`: `t' = t_hat + z(1-eps)·se(t_hat)`. The standard error comes from the
**delta method** applied to `t = -b/w`, using the **inverse penalized Hessian**
of the logistic fit as the parameter covariance. Using the *penalized* Hessian
(rather than the bare Fisher information `XᵀWX`) is deliberate: under separable
data `XᵀWX` is singular and `se` diverges; the penalized version stays finite and
coincides with the bare Fisher information as `lambda → 0`. Documented in
`internal/policy/NOTES.md §4`.

### 4. Which error rate the guarantee bounds — marginal, not conditional

vCache bounds the **marginal** error rate `FP/n` (bad cache hits over *all*
requests), which is exactly the benchmark's metric. It does **not** bound the
conditional rate `FP/(TP+FP)` (bad hits among served hits), which is naturally
higher. Conflating the two was a real bug in the first draft of our own go/no-go
test. Full argument in `NOTES.md §6`.

---

## Cold-start design (our addition — not in the paper)

The logistic fit is undefined at zero observations and diverges under perfectly
separable data. Two guards, **both strictly more conservative than the paper's
bare formula, so neither can weaken the guarantee**:

1. **Force EXPLORE when `len(O) < n_min`.** A fit on a handful of points is
   noise.
2. **L2-regularize the fit** to keep `gamma` finite; a tiny ridge on the bias
   keeps the 2×2 Hessian invertible.

### n_min sensitivity (chosen value: **n_min = 3**, held constant everywhere)

`n_min` trades hit rate against cold-start caution. We picked it **once**, from a
diagnostic sweep, and hold it constant across every run and dataset — it is
never tuned per run, and the go/no-go gate protects any choice.

**Why 3:** it is the smallest floor that keeps the two-parameter logistic fit
(`w`, `b`) *over-determined* — `n_min = 2` exactly determines the fit
(degenerate/interpolating), while `n_min = 3` is a genuine fit — and it
substantially reduces the cold-start conservatism of the original `n_min = 5`.

Diagnostic (SearchQueries, N=40k, seed=42, verified arm hit rate; **every cell
respects its δ**):

| δ | n_min=2 | **n_min=3 (chosen)** | n_min=5 |
|---|---------|----------------------|---------|
| 0.01 | 0.0046 | 0.0033 | 0.0018 |
| 0.02 | 0.0095 | 0.0069 | 0.0037 |
| 0.03 | 0.0138 | 0.0101 | 0.0056 |
| 0.05 | 0.0230 | 0.0179 | 0.0094 |

Realized FP/n stayed ≤ δ for all three n_min values — cold-start is a
hit-rate/caution knob, not a safety knob (safety is the band + the gate).

---

## The systems gaps semblance closes (the paper leaves these open)

1. **Bounded memory / eviction.** Per-shard LRU with a capacity bound
   (`internal/cache`). The paper assumes an unbounded store.
2. **Concurrency.** A sharded lock makes the store safe under concurrent load;
   `Nearest` returns copied snapshots so readers never touch a slice a writer
   mutates. Verified by a `-race` test. The paper is a single-threaded research
   library.
3. **Cold start.** The n_min + L2 design above.
4. **Refit cost.** The logistic fit is cheap (2 params, Newton/IRLS, a few
   iterations) and observations are bounded (below), so re-fitting per request is
   affordable; the gateway can cache the fit and refit on new observations.
5. **Observation-set bounding.** Reservoir sampling caps observations per entry
   (the benchmark uses an equivalent deterministic window). Unbounded histories
   would grow without limit for hot entries.
6. **Correctness scoping / bucketing.** Semantic matching applies only to the
   final user turn, inside an **exact-match bucket** keyed on
   `sha256(model | system prompt | temperature | top_p | prior messages)`. Two
   requests that differ in any of these can never match — the clearest
   correctness signal in the repo.
7. **Non-cacheable bypass.** Requests that cannot be safely cached bypass the
   cache and increment a counter: `stream=true`, any `tools`/`functions`,
   `n > 1`, or `temperature` above a configurable ceiling (default 0.3).

---

## Metrics & budgets

- **`/metrics` is unauthenticated**, like `/healthz`. Prometheus scrapers
  typically run without app credentials, and the endpoint exposes only
  operational aggregates (counts, histograms, costs) — no prompt or response
  content. In a hardened deployment it would be firewalled to the monitoring
  network or scraped via a sidecar; the code keeps it off the authenticated
  `/v1` subtree deliberately.

- **No live `error_rate_realized` — on purpose, and it is a strength.** In
  production you cannot compute the true `FP/n`: on an exploit you serve the
  cached answer and never call the model, so you never learn whether it was
  wrong. Realized error is a **benchmark-only** quantity (it needs ground
  truth). Exposing a fabricated live error rate would be dishonest. Instead we
  expose the configured **delta target** plus honestly-labeled **observable
  proxies**: the judge-observed `c=0` rate among explores
  (`explores_incorrect_total / explores_labeled_total`) and the hit rate
  (derivable from `requests_total`). The entire value of the guarantee is that
  it **bounds an error you cannot otherwise observe live** — the missing metric
  is the point, not a gap.

- **Cost is read from real usage, never estimated.** `spent` and `embedding`
  costs come from the backend/embedding `usage` fields. The `saved` cost on a
  hit is explicitly *modeled*: the served cached answer's own token usage priced
  at the request's model — the honest stand-in for "the call that didn't
  happen." It is not billed to any budget (no real spend occurred). The price
  table carries a committed `retrieved_date` so staleness is visible.

- **Per-key budgets** enforce a spend ceiling (cents) and return HTTP 429 in the
  OpenAI error envelope when exceeded; `budget_remaining_cents{key}` is exported.
  In open mode (no keys) spend is billed to a shared `anonymous` bucket.

## Streaming timeout (to revisit)

The upstream request carries a single total deadline (`BackendTimeout`). For a
streamed response this bounds **total stream duration**, so a generation longer
than the deadline would be cut mid-stream. Acceptable for now; the right fix is a
first-byte / idle timeout for the streaming path rather than a total deadline.

---

## Benchmark decisions (`bench/`)

- **Replayed vectors, not a live embedder.** The harness feeds the dataset's
  precomputed `emb_gte` (1024-dim) directly into the cache and policy, so results
  are deterministic and reproducible from committed vectors.
- **Labeling is ground-truth `id_set` equality, not the LLM judge.** On a replay
  miss the observation label is `neighbor.id_set == query.id_set`. The judge is
  bypassed entirely in the benchmark.
- **Per-arm insertion.** The verified arm uses guarded insert (Algorithm 1); the
  static baseline inserts on every miss (standard semantic-cache behavior). Each
  method is measured end-to-end, as the paper compares them.
- **Purpose-built parallel store.** The harness reuses `internal/policy` verbatim
  (the load-bearing statistics) but uses its own store with a parallel
  nearest-neighbor scan, because `internal/cache` scans a bucket single-threaded
  and the replay is O(n²). The parallel argmax combines chunk results in index
  order, so it is bit-identical to a serial scan.
- **95% Wilson intervals**, seeded RNG (recorded), because the policy is
  randomized.

### Dataset label validation (all via `tools/prep/extract.py`)

- **Combo** — dev-loop only. 26 impure `id_set` groups, all **negative** ids: a
  singleton/marker convention that lumps unrelated prompts. Not used for any
  published number.
- **SearchQueries** — secondary result. Responses are a **constant placeholder**,
  so response-purity is meaningless (trivially pure); scoring relies on `id_set`,
  validated structurally (11,240 multi-member clusters). A **sparse-repeat**
  workload: only ~20% of rows have a cached same-cluster sibling. Disclosed
  plainly — the verified arm is *safe but conservative* here, while an untuned
  static threshold reaches **37.6% error** (T=0.80).
- **LmArena** — headline. Responses are **real free-text** (60,408 distinct of
  63,796 rows), so exact-match response-purity is also uninformative — but for
  the opposite reason (surface-form variation, not bad clustering). The `id_set`
  structure is excellent: **3,398 multi-member clusters, 99.8% of rows in a
  cluster, 0 negative ids**; spot-checked as genuine paraphrases. A dense-repeat
  workload, which is why it is the headline.
