package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestWilson(t *testing.T) {
	const z = 1.959963985
	// x=50/100 → center 0.5, half-width ≈ 0.0961 (well-known value).
	lo, hi, p := wilson(50, 100, z)
	if math.Abs(p-0.5) > 1e-9 {
		t.Errorf("p = %v, want 0.5", p)
	}
	if math.Abs(lo-0.4038) > 1e-3 || math.Abs(hi-0.5962) > 1e-3 {
		t.Errorf("interval = [%.4f, %.4f], want ~[0.4038, 0.5962]", lo, hi)
	}
	// Zero successes: point estimate 0, lower bound 0, upper bound > 0.
	lo0, hi0, p0 := wilson(0, 1000, z)
	if p0 != 0 || lo0 < 0 || hi0 <= 0 {
		t.Errorf("zero-success interval = [%.5f, %.5f], p=%v", lo0, hi0, p0)
	}
}

// TestNearestParallelMatchesSerial: the parallel scan must return the exact same
// argmax as a serial scan (same index and similarity), for reproducibility.
func TestNearestParallelMatchesSerial(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const n, d = 5000, 16 // n > parThreshold to trigger the parallel path
	entries := make([]entry, n)
	for i := range entries {
		v := make([]float32, d)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		entries[i] = entry{vec: normalize(v)}
	}
	q := normalize([]float32{1, 2, -3, 4, 0, 1, -1, 2, 3, -2, 1, 0, -1, 1, 2, -3})

	si, ss := nearest(q, entries, 1)
	pi, ps := nearest(q, entries, 8)
	if si != pi || math.Abs(ss-ps) > 1e-12 {
		t.Errorf("parallel (%d, %.6f) != serial (%d, %.6f)", pi, ps, si, ss)
	}
}

// TestScoringTP: an identical repeat with matching id_set, above threshold, is a
// true-positive exploit.
func TestScoringTP(t *testing.T) {
	records := []record{
		{IDSet: 1, Embedding: []float32{1, 0}},
		{IDSet: 1, Embedding: []float32{1, 0}}, // identical → sim 1.0
		{IDSet: 2, Embedding: []float32{0, 1}}, // orthogonal → miss
	}
	r := runArm(records, "static", 0.99, 42, 0, 1, 5)
	if r.TP != 1 || r.FP != 0 {
		t.Errorf("TP=%d FP=%d, want TP=1 FP=0", r.TP, r.FP)
	}
	if r.Exploits != 1 || r.Explores != 2 {
		t.Errorf("exploits=%d explores=%d, want 1 and 2", r.Exploits, r.Explores)
	}
	if math.Abs(r.ErrorRate-0) > 1e-9 || math.Abs(r.HitRate-1.0/3.0) > 1e-9 {
		t.Errorf("err=%v hit=%v, want 0 and 1/3", r.ErrorRate, r.HitRate)
	}
}

// TestScoringFP: an identical repeat with a DIFFERENT id_set, above threshold,
// is a false-positive exploit.
func TestScoringFP(t *testing.T) {
	records := []record{
		{IDSet: 1, Embedding: []float32{1, 0}},
		{IDSet: 2, Embedding: []float32{1, 0}}, // identical vector, different label
	}
	r := runArm(records, "static", 0.99, 42, 0, 1, 5)
	if r.TP != 0 || r.FP != 1 {
		t.Errorf("TP=%d FP=%d, want TP=0 FP=1", r.TP, r.FP)
	}
	if math.Abs(r.ErrorRate-0.5) > 1e-9 {
		t.Errorf("err=%v, want 0.5", r.ErrorRate)
	}
}

// TestInvariants: exploits = TP+FP, and explores+exploits = n, on a mixed run.
func TestInvariants(t *testing.T) {
	records := makeClusteredRecords(50, 10, 32, 0.02, 7)
	for _, arm := range []string{"static", "verified"} {
		r := runArm(records, arm, 0.9, 42, 0, 4, 5)
		if r.Exploits != r.TP+r.FP {
			t.Errorf("%s: exploits %d != TP+FP %d", arm, r.Exploits, r.TP+r.FP)
		}
		if r.Explores+r.Exploits != r.N {
			t.Errorf("%s: explores+exploits %d != n %d", arm, r.Explores+r.Exploits, r.N)
		}
	}
}

// TestVerifiedRespectsDeltaSynthetic is a fast go/no-go on a synthetic clustered
// dataset: the verified arm's realized FP/n must not exceed delta.
func TestVerifiedRespectsDeltaSynthetic(t *testing.T) {
	// 200 clusters x 25 members, tight intra-cluster similarity.
	records := makeClusteredRecords(200, 25, 64, 0.015, 123)
	for _, delta := range []float64{0.01, 0.02, 0.05} {
		r := runArm(records, "verified", delta, 42, 0, 4, 5)
		// Allow a tiny Monte-Carlo margin (the harness itself uses the strict
		// gate on the real dataset).
		if r.ErrorRate > delta+0.005 {
			t.Errorf("delta=%.2f: FP/n=%.5f exceeds budget (hit=%.4f)", delta, r.ErrorRate, r.HitRate)
		}
		t.Logf("delta=%.2f: FP/n=%.5f hit=%.4f entries=%d", delta, r.ErrorRate, r.HitRate, r.EntriesFinal)
	}
}

// makeClusteredRecords builds a shuffled dataset of k clusters, each with m
// members formed by adding small Gaussian noise to a random center. Members of a
// cluster share an id_set and are near in cosine; different clusters differ.
func makeClusteredRecords(k, m, d int, noise float64, seed int64) []record {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, k)
	for c := range centers {
		v := make([]float32, d)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		centers[c] = v
	}
	var recs []record
	for c := 0; c < k; c++ {
		for i := 0; i < m; i++ {
			v := make([]float32, d)
			for j := range v {
				v[j] = centers[c][j] + float32(rng.NormFloat64()*noise)
			}
			recs = append(recs, record{IDSet: int64(c), Embedding: v})
		}
	}
	rng.Shuffle(len(recs), func(i, j int) { recs[i], recs[j] = recs[j], recs[i] })
	return recs
}
