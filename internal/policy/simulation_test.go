package policy

import (
	"math"
	"math/rand"
	"testing"
)

// TestRealizedErrorWithinBudget is the go/no-go gate for the whole project.
//
// We simulate a stream of queries against a single cache entry whose true
// correctness curve is a known sigmoid L(s; trueT, trueGamma). For each query:
//
//   - draw similarity s and the latent correctness c ~ Bernoulli(L(s));
//   - ask the policy to EXPLORE or EXPLOIT (learning only on EXPLORE, per the
//     brief);
//   - if we EXPLOIT and the cached answer was actually wrong (c=0), that is a
//     false-positive cache hit that the user experiences as an error.
//
// WHICH error rate does vCache bound? The marginal one: FP / n, the fraction of
// ALL requests that are erroneous cache hits. This is the paper's guarantee and
// exactly the benchmark's definition (error rate = FP/n). It is NOT the
// conditional rate FP/(TP+FP) = errors among hits, which is naturally higher.
//
// Why marginal: the decision sets (1-tau) = delta/(1-alpha) with
// alpha = (1-eps)*L(s, t', gamma). Because t' is a pessimistic upper bound on
// the true threshold, L(s,t',gamma) <= L_true(s), so alpha <= L_true(s), and
// therefore the per-query probability of an erroneous exploit is
//
//	(1 - tau(s)) * (1 - L_true(s)) = delta * (1 - L_true(s)) / (1 - alpha) <= delta.
//
// Summing over queries gives FP/n <= delta. See NOTES.md for the full argument
// and for why the conditional rate is the wrong thing to bound here.
//
// If FP/n exceeds delta, the statistics are wrong and nothing downstream
// matters — stop and debug.
func TestRealizedErrorWithinBudget(t *testing.T) {
	type scenario struct {
		name      string
		trueT     float64
		trueGamma float64
	}
	scenarios := []scenario{
		{"moderate", 0.80, 25},
		{"sharp-high", 0.90, 45},
	}
	deltas := []float64{0.01, 0.02, 0.05}

	const (
		n         = 80000 // queries per run
		windowCap = 200   // cap on the observation set (reservoir stand-in)
	)

	for _, sc := range scenarios {
		for _, delta := range deltas {
			t.Run(sc.name+"/delta="+ftoa(delta), func(t *testing.T) {
				// Separate RNGs for the environment and the policy coin flip,
				// both seeded, so the whole run is reproducible.
				envRng := rand.New(rand.NewSource(12345))
				polRng := rand.New(rand.NewSource(999))
				pol := NewPolicy(delta, polRng)

				window := make([]Observation, 0, windowCap+1)
				var exploits, errors, explores int

				for i := 0; i < n; i++ {
					s := 0.6 + 0.4*envRng.Float64() // Uniform(0.6, 1.0)
					pTrue := L(s, sc.trueT, sc.trueGamma)
					cTrue := envRng.Float64() < pTrue

					d := pol.Decide(window, s)
					if d.Explore {
						explores++
						window = append(window, Observation{S: s, C: cTrue})
						if len(window) > windowCap {
							// keep most recent windowCap (simple sliding window)
							window = window[len(window)-windowCap:]
						}
					} else {
						exploits++
						if !cTrue {
							errors++
						}
					}
				}

				if exploits < 1000 {
					t.Fatalf("too few exploits (%d) to judge the guarantee — test is vacuous", exploits)
				}

				// The guarantee: marginal error rate FP/n <= delta.
				marginalErr := float64(errors) / float64(n)
				// Informational: conditional rate among hits, and the hit rate.
				conditionalErr := float64(errors) / float64(exploits)
				hitRate := float64(exploits) / float64(n)

				// Monte-Carlo tolerance on the marginal rate: if the true
				// per-query erroneous-exploit probability is <= delta, the
				// empirical mean over n draws sits within a few standard errors
				// (each query's variance is at most delta). 4 sigma + a hair
				// guards against flaky failures while still catching a genuinely
				// broken policy (which lands far above delta).
				margin := 4*math.Sqrt(delta/float64(n)) + 1e-4
				bound := delta + margin

				t.Logf("hitRate=%.3f  FP/n=%.5f (bound %.5f, delta=%.3f)  err|hit=%.4f",
					hitRate, marginalErr, bound, delta, conditionalErr)

				if marginalErr > bound {
					t.Errorf("REALIZED ERROR EXCEEDS BUDGET: FP/n=%.5f > %.5f (delta=%.3f). "+
						"The verified-caching guarantee is violated.", marginalErr, bound, delta)
				}
			})
		}
	}
}

// ftoa formats a delta for a subtest name without pulling in fmt in a hot path.
func ftoa(f float64) string {
	switch f {
	case 0.01:
		return "0.01"
	case 0.02:
		return "0.02"
	case 0.05:
		return "0.05"
	default:
		return "x"
	}
}
