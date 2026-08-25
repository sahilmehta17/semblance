package main

import (
	"math"
	"math/rand"
	"sync"

	"github.com/sahilmehta17/semblance/internal/policy"
)

// record is one replayed query from the extracted JSONL. The prompt text is not
// needed for scoring (labels come from id_set), so it is ignored on decode.
type record struct {
	IDSet     int64     `json:"id_set"`
	Embedding []float32 `json:"embedding"`
}

// entry is a cached prompt in the replay store: its normalized vector, its
// ground-truth id_set (the label), and — for the verified arm — the observation
// set the policy fits.
type entry struct {
	vec   []float32
	idSet int64
	obs   []policy.Observation
}

// obsWindow caps observations per entry with a deterministic sliding window
// (keep most-recent). The shipped cache uses reservoir sampling; either is a
// valid observation subset and the guarantee holds for both. A window keeps the
// benchmark fully deterministic (no extra RNG) and bounds the per-query fit cost.
const obsWindow = 1024

// parThreshold is the entry count above which the nearest scan fans out across
// worker goroutines. Below it the goroutine overhead is not worth it.
const parThreshold = 4096

// runResult holds one arm/parameter run's scored outcome.
type runResult struct {
	Arm          string  `json:"arm"`   // "static" | "verified"
	Param        float64 `json:"param"` // threshold (static) or delta (verified)
	Delta        float64 `json:"delta,omitempty"`
	N            int     `json:"n"`
	TP           int     `json:"tp"`
	FP           int     `json:"fp"`
	Exploits     int     `json:"exploits"`
	Explores     int     `json:"explores"`
	HitRate      float64 `json:"hit_rate"`
	HitLo        float64 `json:"hit_ci_lo"`
	HitHi        float64 `json:"hit_ci_hi"`
	ErrorRate    float64 `json:"error_rate"`
	ErrorLo      float64 `json:"error_ci_lo"`
	ErrorHi      float64 `json:"error_ci_hi"`
	EntriesFinal int     `json:"entries_final"`
	NMin         int     `json:"nmin,omitempty"`         // verified arm only
	WithinDelta  bool    `json:"within_delta,omitempty"` // verified arm only
}

// runArm replays every record in order through one arm.
//
// Decision rules:
//   - static:   EXPLOIT iff a neighbor exists and similarity >= threshold.
//   - verified: EXPLOIT iff policy.Decide(neighbor.obs, similarity) says exploit.
//
// Labeling / insertion (mirrors the shipped cache):
//   - The observation label c is GROUND-TRUTH id_set equality (neighbor.idSet ==
//     query.idSet). The judge is bypassed entirely.
//   - Learning happens only on EXPLORE, and only for the verified arm.
//   - Guarded insert (Algorithm 1): the verified arm inserts a new entry only
//     when the neighbor was wrong (c == false). The static baseline inserts on
//     every miss (standard semantic-cache behavior). A cold miss (no neighbor)
//     inserts unconditionally in both arms.
//
// Scoring (authors' convention): on an EXPLOIT, TP if the matched entry's id_set
// equals the query's, else FP. error rate = FP/n, hit rate = (TP+FP)/n.
func runArm(records []record, arm string, param float64, seed int64, capacity, workers, nmin int) runResult {
	var pol *policy.Policy
	if arm == "verified" {
		// Seeded RNG so the randomized policy is reproducible. Draws happen in
		// query order (the scan is parallel but read-only), so the seed fully
		// determines the outcome.
		pol = policy.NewPolicy(param, rand.New(rand.NewSource(seed)))
		// nmin is the cold-start floor (force-explore below this many
		// observations). It is chosen ONCE and held constant across all runs;
		// the go/no-go gate protects against a value that would breach delta.
		pol.NMin = nmin
	}

	entries := make([]entry, 0, min(len(records), capacityOrCap(capacity, len(records))))
	var tp, fp, explores int

	for i := range records {
		q := normalize(records[i].Embedding)
		qid := records[i].IDSet

		bi, sim := nearest(q, entries, workers)
		haveNeighbor := bi >= 0

		exploit := false
		if haveNeighbor {
			if arm == "static" {
				exploit = sim >= param
			} else {
				exploit = !pol.Decide(entries[bi].obs, sim).Explore
			}
		}

		if exploit {
			if entries[bi].idSet == qid {
				tp++
			} else {
				fp++
			}
			continue // EXPLOIT serves cached; no learning
		}

		// EXPLORE (miss).
		explores++
		if haveNeighbor {
			c := entries[bi].idSet == qid
			if arm == "verified" {
				entries[bi].obs = appendObs(entries[bi].obs, policy.Observation{S: sim, C: c})
				if !c { // guarded insert: only when the neighbor was wrong
					entries = insertEntry(entries, q, qid, capacity)
				}
			} else {
				entries = insertEntry(entries, q, qid, capacity) // static: always insert
			}
		} else {
			entries = insertEntry(entries, q, qid, capacity) // cold miss
		}
	}

	n := len(records)
	exploits := tp + fp
	const z = 1.959963985 // 95%
	res := runResult{
		Arm:          arm,
		Param:        param,
		N:            n,
		TP:           tp,
		FP:           fp,
		Exploits:     exploits,
		Explores:     explores,
		EntriesFinal: len(entries),
	}
	res.HitLo, res.HitHi, res.HitRate = wilson(exploits, n, z)
	res.ErrorLo, res.ErrorHi, res.ErrorRate = wilson(fp, n, z)
	if arm == "verified" {
		res.Delta = param
		res.NMin = nmin
		res.WithinDelta = res.ErrorRate <= param
	}
	return res
}

// nearest returns the index and cosine similarity of the highest-similarity
// entry, or (-1, 0) if empty. The scan is parallelized across workers for large
// entry sets; results are combined in index order so the argmax is identical to
// a serial scan (first index achieving the max) — keeping runs reproducible.
func nearest(q []float32, entries []entry, workers int) (int, float64) {
	n := len(entries)
	if n == 0 {
		return -1, 0
	}
	if n < parThreshold || workers <= 1 {
		return nearestRange(q, entries, 0, n)
	}

	type res struct {
		idx int
		sim float64
	}
	results := make([]res, workers)
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := lo + chunk
		if lo >= n {
			results[w] = res{-1, math.Inf(-1)}
			continue
		}
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			idx, sim := nearestRange(q, entries, lo, hi)
			results[w] = res{idx, sim}
		}(w, lo, hi)
	}
	wg.Wait()

	bestIdx, bestSim := -1, math.Inf(-1)
	for _, r := range results { // in worker (=index) order → earliest max wins
		if r.idx >= 0 && r.sim > bestSim {
			bestSim, bestIdx = r.sim, r.idx
		}
	}
	return bestIdx, bestSim
}

// nearestRange scans entries[lo:hi] and returns the first index (>= lo) with the
// maximum dot product against q.
func nearestRange(q []float32, entries []entry, lo, hi int) (int, float64) {
	bestIdx, bestSim := -1, math.Inf(-1)
	for k := lo; k < hi; k++ {
		s := dot(q, entries[k].vec)
		if s > bestSim {
			bestSim, bestIdx = s, k
		}
	}
	return bestIdx, bestSim
}

// insertEntry appends a new entry, evicting the oldest if at capacity (capacity
// <= 0 means unbounded — the paper's no-eviction setup).
func insertEntry(entries []entry, vec []float32, idSet int64, capacity int) []entry {
	if capacity > 0 && len(entries) >= capacity {
		copy(entries, entries[1:]) // drop oldest (FIFO)
		entries = entries[:len(entries)-1]
	}
	return append(entries, entry{vec: vec, idSet: idSet})
}

// appendObs keeps the most-recent obsWindow observations.
func appendObs(obs []policy.Observation, o policy.Observation) []policy.Observation {
	if len(obs) >= obsWindow {
		copy(obs, obs[1:])
		obs[len(obs)-1] = o
		return obs
	}
	return append(obs, o)
}

// normalize returns a unit-length copy so cosine similarity is a dot product.
func normalize(vec []float32) []float32 {
	out := make([]float32, len(vec))
	var norm float64
	for _, x := range vec {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		return out
	}
	inv := float32(1 / math.Sqrt(norm))
	for i, x := range vec {
		out[i] = x * inv
	}
	return out
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// wilson returns the (lo, hi) 95%-ish score interval and the point estimate for
// x successes in n trials. z sets the confidence level. The Wilson interval is
// used (rather than normal-approximation) because it stays valid for the very
// small proportions we expect (error rates near 0.01).
func wilson(x, n int, z float64) (lo, hi, p float64) {
	if n == 0 {
		return 0, 0, 0
	}
	nn := float64(n)
	p = float64(x) / nn
	denom := 1 + z*z/nn
	center := (p + z*z/(2*nn)) / denom
	half := z * math.Sqrt(p*(1-p)/nn+z*z/(4*nn*nn)) / denom
	return center - half, center + half, p
}

func capacityOrCap(capacity, n int) int {
	if capacity > 0 && capacity < n {
		return capacity
	}
	return n
}
