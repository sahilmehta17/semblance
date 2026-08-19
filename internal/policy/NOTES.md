# `internal/policy` — the statistics, in plain language

This package is the one an interviewer will probe, so this document explains
every moving part from first principles. If you can talk through this file, you
can defend the package. It is a clean-room implementation of the decision rule
in **vCache** (Schroeder et al., *Verified Semantic Prompt Caching*, ICLR 2026,
arXiv:2502.03771); no vCache source was consulted. Where the paper is ambiguous
we picked a defensible reading and flagged it as an **interpretation** below.

---

## 1. The problem in one sentence

We have a cached answer for some prompt. A new, *similar* prompt arrives. Should
we reuse the cached answer (fast, cheap, but maybe wrong) or call the model
again (slow, expensive, correct)? We want to reuse it **only when we can bound
the chance of being wrong by a budget `delta` the user chooses.**

The naive approach is a fixed similarity threshold: "reuse if cosine ≥ 0.95."
The problem is that the right threshold differs per entry and you never know if
0.95 is safe. vCache replaces the fixed threshold with a *learned, per-entry*
model of correctness plus a *statistical* decision that respects `delta`.

---

## 2. What each cache entry learns

For one cache entry we keep an **observation set** `O = {(s_i, c_i)}`:

- `s_i` — cosine similarity between a later query and this entry.
- `c_i` — 1 if this entry's stored answer *would have been correct* for that
  query, else 0. (In production we only get this label when we EXPLORE — call
  the backend — and compare; see §7.)

We then model the probability the entry is correct as a function of similarity:

```
L(s; t, gamma) = sigmoid(gamma * (s - t)) = 1 / (1 + e^(-gamma (s - t)))
```

- `t` is the **threshold**: the similarity at which correctness is a coin flip
  (probability 0.5).
- `gamma` is the **steepness**: how fast correctness rises as `s` climbs past
  `t`. Large `gamma` = a nearly hard cutoff; small `gamma` = a gentle ramp.

This is just logistic regression of `c` on `s`. In the usual
`P = sigmoid(w*s + b)` form, `gamma = w` and `t = -b/w`. That identity is the
whole reason we can fit it with a textbook logistic regression and then read off
`(t, gamma)`. (`logistic.go`)

---

## 3. Fitting the curve (`logistic.go`)

We minimize the **regularized negative log-likelihood** (binary cross-entropy):

```
J(w,b) = -sum_i [ c_i log p_i + (1-c_i) log(1-p_i) ] + (lambda/2) w^2 + (biasRidge/2) b^2
         where p_i = sigmoid(w s_i + b)
```

We solve it with **Newton's method / IRLS** (iteratively reweighted least
squares): each step computes the gradient and the 2×2 Hessian of `J` and takes a
Newton step `theta -= H^{-1} grad`. Because there are only two parameters, the
"matrix algebra" is a 2×2 inverse we do by hand. It converges in a handful of
iterations.

**Note on the paper's Eq. 10.** The paper writes an `arg min` over a
log-*likelihood*; taken literally that maximizes error. It is a typo — the
intended objective is standard cross-entropy *minimization*, which is what we
implement.

**Why regularize (this is our design, not the paper's).** With *perfectly
separable* data — every low-`s` query wrong, every high-`s` query right — the
unregularized fit drives `gamma → ∞` (an infinitely sharp, infinitely confident
step). That is numerically explosive and statistically a lie. The ridge term
`(lambda/2) w^2` keeps `gamma` finite. The tiny `biasRidge` on `b` makes the
objective strongly convex so the Hessian is always invertible; it is small
enough not to move `t` meaningfully. **Both make the model *less* confident than
the bare MLE, so neither can weaken the guarantee** (see §6, §8).

---

## 4. How sure are we about `t`? (`band.go`)

The fitted `t_hat` is an estimate from finite, noisy data. We need to know *how
uncertain* it is. Two pieces:

1. **Covariance of `(w, b)`.** The inverse of the Hessian at the optimum is the
   estimator's covariance `Sigma`. (This is the inverse *Fisher information* of
   the fit.)

   *Interpretation:* we use the **penalized** Hessian `X^T W X + R` rather than
   the bare Fisher information `X^T W X`. Under separable data the bare version
   is singular and the standard error blows up to infinity; the penalized
   version stays finite and equals the bare version as `lambda → 0`. This is a
   deliberate, conservative reading of the paper's "inverse Fisher information."

2. **Standard error of `t` via the delta method.** `t = -b/w` is a nonlinear
   function of the parameters, so we propagate the covariance through its
   gradient:

   ```
   Var(t) ≈ (∂t/∂w, ∂t/∂b) · Sigma · (∂t/∂w, ∂t/∂b)^T
   ∂t/∂w = b / w^2,   ∂t/∂b = -1 / w
   se(t) = sqrt(Var(t))
   ```

`band.go` also contains `normalQuantile`, the inverse standard-normal CDF (the
"probit", `z` such that `Phi(z) = p`). Go's stdlib has the normal CDF's cousins
(`math.Erf`) but not the inverse, and the brief forbids a stats library, so we
implement Acklam's rational approximation (error < 1.15e-9).

---

## 5. The decision (`decide.go`) — and why it's randomized

Here is the part that makes verified caching *verified*. We do **not** compare
`s` to `t_hat`. Instead we compute an **exploration probability `tau`**, flip a
coin `u ~ Uniform(0,1)`, and:

- `u ≤ tau`  → **EXPLORE** (treat as a miss, call the backend, learn).
- `u > tau`  → **EXPLOIT** (serve the cached answer).

Randomization is what lets us *prove* a bound. A hard threshold gives you no
knob to trade a little more exploration for a guaranteed error ceiling; a
probability does. `tau` is that knob, set per request so the ceiling always
holds.

**Computing `tau` (the `tauHat` grid).** For each `eps` on a 50-point grid in
`(0,1)`:

```
t'    = t_hat + z(1-eps) * se(t_hat)      # pessimistic UPPER bound on t
alpha = (1 - eps) * L(s, t', gamma_hat)   # high-confidence LOWER bound on correctness
tau   = 1 - delta / (1 - alpha)
```

and take `tau_hat = min over the grid`, clamped to `[0,1]`.

Reading it:

- Pushing `t` *up* to `t'` is pessimistic: a higher threshold means the entry is
  *less* likely to be right at this `s`. `z(1-eps)` says how many standard
  errors of pessimism (bigger for smaller `eps`).
- `alpha` is a lower bound on the true correctness probability that already
  discounts the `eps` chance the band is wrong (the `(1-eps)` factor).
- `tau = 1 - delta/(1-alpha)`. When the entry looks reliable (`alpha` near 1),
  `1-alpha` is tiny, `delta/(1-alpha)` is large, `tau` clamps to 0 → **exploit**.
  When it looks unreliable (`alpha` near 0), `tau ≈ delta` short of 1 → **mostly
  explore**.
- We **minimize over `eps`** because each `eps` gives a *valid* bound; the
  smallest `tau` is the least exploration (highest hit rate) that is still safe.

**Cold start (our design).** Two guards, both strictly conservative:
- If `len(O) < n_min` (default 5), force EXPLORE. A logistic fit on a few points
  is noise.
- If the fit is degenerate (non-positive slope, non-finite band), force EXPLORE.

---

## 6. Which error rate is actually bounded? (read this twice)

**vCache bounds the *marginal* error rate `FP / n`** — the fraction of *all*
requests that are erroneous cache hits. It does **not** bound the *conditional*
rate `FP / (TP+FP)` — errors among the requests we chose to serve from cache.
The benchmark uses the same marginal definition (`error rate = FP/n`).

Why marginal falls out of the formula. Because `t'` is a pessimistic upper bound
on the true `t`, we have `L(s,t',gamma) ≤ L_true(s)`, hence `alpha ≤ L_true(s)`.
So the per-request probability of an erroneous exploit is

```
(1 - tau(s)) * (1 - L_true(s)) = delta * (1 - L_true(s)) / (1 - alpha) ≤ delta.
```

Sum over requests → `FP/n ≤ delta`. The `min`-over-`eps` keeps this tight.

**Why the conditional rate is higher, and why that's fine.** At a low-similarity
entry where the cached answer is almost always wrong, the safe thing is to serve
it very rarely — about a `delta` fraction of the time. But *when* we do serve it,
it's usually wrong, so the conditional rate there is near 100%. Averaged over the
whole stream those rare bad hits are still only a `delta` slice of traffic. Our
simulation shows exactly this: `FP/n` sits just under `delta` while `err|hit`
runs 12–42%. If a stakeholder asks "how often is a *served* answer wrong?", the
honest answer is "the conditional rate can exceed `delta`; what `delta` bounds is
the share of *all* traffic that is a bad hit." Confusing the two is the classic
mistake — and it was a bug in the first draft of our own go/no-go test.

---

## 7. Learning loop

- We append `(s, c)` to the nearest neighbor's `O` **only on an EXPLORE**, since
  only then did we actually call the backend and learn the true label. Exploits
  teach us nothing new (we didn't check).
- Labeling `c`: exact string match for short responses; an async LLM-judge for
  long ones (that lives in `internal/judge`, off the hot path).
- Observation sets are capped (reservoir sampling in the cache layer) so fitting
  stays cheap; the simulation approximates this with a sliding window.

---

## 8. What would break the guarantee

Concrete failure modes to keep in mind (and test against):

1. **An under-estimated band.** If `se(t)` is too small (e.g. we used the bare,
   over-confident Fisher information and it collapsed under separable data), `t'`
   isn't pessimistic enough, `alpha` overshoots `L_true`, and errors exceed
   `delta`. Our penalized covariance + cold-start guard against this.
2. **A mis-specified curve.** The bound assumes correctness really is monotone in
   similarity and roughly logistic. If a *higher*-similarity neighbor were
   *more* wrong (non-monotone), the model is wrong and the guarantee is void. We
   reject non-positive-slope fits for this reason.
3. **Learning on exploits.** If we appended labels from exploited requests
   (which we never actually verified), `O` would be poisoned with guesses.
4. **Matching across buckets.** The whole thing is per-entry within an
   exact-match bucket keyed on model + system prompt + params. Let two different
   system prompts share an entry and `s` no longer means what the model thinks.
5. **The wrong error metric.** Reporting the conditional rate as if it were the
   guaranteed one (see §6) doesn't break the math — it breaks the *claim*.

---

## 9. Map of the files

| File | Responsibility |
|---|---|
| `logistic.go` | `Observation`, `Model`, the L2-regularized IRLS fit, `L`/`sigmoid`. |
| `band.go` | Delta-method `se(t)`; `normalQuantile` (inverse normal CDF). |
| `decide.go` | `Policy`, the `eps`-grid `tau_hat`, cold-start, randomized decision. |
| `*_test.go` | Unit tests + the **go/no-go simulation** (`FP/n ≤ delta`). |

The go/no-go test is the gate: if realized `FP/n` ever exceeds `delta`, the
statistics are wrong and nothing built on top of this matters.
