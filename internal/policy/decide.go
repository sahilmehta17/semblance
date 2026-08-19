package policy

import "math"

// Rand is the minimal random-number source the policy needs. Injecting it
// (rather than calling the global rand package) is what makes decisions
// deterministic in tests: a test can supply a fixed or seeded source and get
// reproducible EXPLORE/EXPLOIT outcomes.
type Rand interface {
	// Float64 returns a pseudo-random number in [0.0, 1.0).
	Float64() float64
}

// Policy holds the configuration for the verified-caching decision rule.
// Delta is the only real hyperparameter; the rest have safe, conservative
// defaults set by NewPolicy.
type Policy struct {
	// Delta is the user's error budget: the target upper bound on the
	// probability of serving an incorrect cached response. Smaller = safer =
	// fewer cache hits.
	Delta float64
	// NMin forces exploration until an entry has at least this many
	// observations. The logistic fit is meaningless on a handful of points, so
	// we refuse to trust it early. Part of our cold-start design (NOTES.md).
	NMin int
	// Lambda is the L2 ridge strength on the slope in the logistic fit.
	Lambda float64
	// GridSize is the number of eps points used when minimizing tau.
	GridSize int
	// Rand is the injected randomness source for the explore/exploit coin flip.
	Rand Rand
}

// NewPolicy returns a Policy with the brief's defaults, overriding only delta
// and the randomness source (the two things a caller must supply).
func NewPolicy(delta float64, r Rand) *Policy {
	return &Policy{
		Delta:    delta,
		NMin:     5,    // force-explore below 5 observations
		Lambda:   1e-2, // light ridge; just enough to keep gamma finite
		GridSize: 50,   // 50-point eps grid, per the brief
		Rand:     r,
	}
}

// Decision is the outcome of the policy for one request.
type Decision struct {
	// Explore is true if the caller should treat this as a cache MISS (call the
	// backend and learn from the result); false means EXPLOIT (serve cached).
	Explore bool
	// Tau is the exploration probability that produced this decision. Exposed
	// so the gateway can surface it in the X-Semblance-Tau response header.
	Tau float64
	// S is the similarity of the query to its nearest cache entry.
	S float64
	// Model is the fitted correctness curve, or a zero (Usable=false) model
	// when the decision was forced by the cold-start rule.
	Model Model
	// Forced is true when exploration was forced (cold start or degenerate
	// fit) rather than chosen by the coin flip.
	Forced bool
}

// Decide runs the full rule for a query whose nearest cache entry has the given
// observation set and cosine similarity s.
//
// Steps (mirroring the brief):
//  1. Cold start: if we have fewer than NMin observations, EXPLORE.
//  2. Fit the logistic correctness curve; if it is degenerate, EXPLORE.
//  3. Compute the exploration probability tau_hat.
//  4. Draw u ~ Uniform(0,1); EXPLORE if u <= tau_hat, else EXPLOIT.
func (p *Policy) Decide(obs []Observation, s float64) Decision {
	if len(obs) < p.NMin {
		return Decision{Explore: true, Tau: 1, S: s, Forced: true}
	}

	m := Fit(obs, p.Lambda)
	if !m.Usable {
		return Decision{Explore: true, Tau: 1, S: s, Model: m, Forced: true}
	}

	tau := p.tauHat(&m, s)
	// A non-finite tau means the band blew up on degenerate data; explore.
	if math.IsNaN(tau) {
		return Decision{Explore: true, Tau: 1, S: s, Model: m, Forced: true}
	}

	u := p.Rand.Float64()
	return Decision{
		Explore: u <= tau,
		Tau:     tau,
		S:       s,
		Model:   m,
	}
}

// tauHat computes the exploration probability for query similarity s, given a
// fitted model. This is the mathematical core of verified caching.
//
// We do not know the true threshold t exactly — we have an estimate t_hat and a
// standard error se(t_hat). For a confidence level (1-eps), the one-sided
// upper confidence bound on the true threshold is
//
//	t' = t_hat + z(1-eps) * se(t_hat)
//
// where z is the standard-normal quantile. Pushing t upward is pessimistic:
// a higher threshold means the entry is LESS likely to be correct at this s.
// Using that worst-case threshold we form
//
//	alpha  = (1 - eps) * L(s, t', gamma_hat)
//	tau    = 1 - delta / (1 - alpha)
//
// and minimize tau over a grid of eps in (0,1). The minimizing eps is the
// confidence level that lets us explore the least while still keeping the
// error bounded by delta; taking the min therefore maximizes the hit rate
// subject to the guarantee. The result is clamped to [0,1].
//
// Intuition for the two ends: at very high similarity, L(s, t', gamma) ≈ 1, so
// 1 - alpha ≈ eps and tau ≈ 1 - delta/eps, which clamps to 0 → serve cached.
// At low similarity, L ≈ 0, so alpha ≈ 0 and tau ≈ delta near 1 → mostly
// explore. The band widens tau whenever the fit is uncertain.
func (p *Policy) tauHat(m *Model, s float64) float64 {
	se := m.seT()
	best := math.Inf(1)

	for k := 0; k < p.GridSize; k++ {
		// eps points sit strictly inside (0,1): 0.01, 0.03, ..., 0.99 for a
		// 50-point grid. Avoiding the exact endpoints keeps z(1-eps) finite.
		eps := (float64(k) + 0.5) / float64(p.GridSize)

		z := normalQuantile(1 - eps)
		tPrime := m.T + z*se
		alpha := (1 - eps) * L(s, tPrime, m.Gamma)

		// alpha < 1 always here (alpha <= (1-eps) < 1), so 1-alpha > 0 and the
		// division is safe.
		tau := 1 - p.Delta/(1-alpha)
		if tau < best {
			best = tau
		}
	}

	return clamp(best, 0, 1)
}

func clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}
