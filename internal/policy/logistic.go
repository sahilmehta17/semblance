// Package policy implements the per-entry statistical decision rule at the
// heart of verified semantic caching.
//
// It is a clean-room implementation of the algorithm described in:
//
//	Schroeder, L. et al. "vCache: Verified Semantic Prompt Caching."
//	ICLR 2026. arXiv:2502.03771. UC Berkeley Sky Computing Lab.
//
// No vCache source code was consulted or copied; everything here is derived
// from the paper's equations. Where the paper is ambiguous, we implement the
// defensible reading and document the interpretation in NOTES.md.
//
// The idea in one paragraph: for a cache entry we keep a set of observations
// (s, c) — the cosine similarity s of a later query to this entry, and whether
// this entry's stored answer would have been correct (c) for that query. We
// fit a logistic curve P(correct | s) = sigmoid(gamma*(s - t)) to those
// observations. Given a new query at similarity s, we do NOT simply compare s
// to a fixed threshold; instead we compute an exploration probability tau that
// accounts for our uncertainty in the fitted threshold t, and flip a biased
// coin. This is what lets us bound the error rate by a user-chosen budget
// delta. The fit lives in this file; the uncertainty band lives in band.go;
// the decision lives in decide.go.
package policy

import "math"

// Observation is a single training point for an entry's correctness curve.
type Observation struct {
	// S is the cosine similarity between the querying prompt and the cache
	// entry. For our normalized embeddings this is in [-1, 1], and in practice
	// clusters in the upper part of [0, 1].
	S float64
	// C is the label: true if the entry's stored response would have been
	// correct for that query. Stored as bool for callers; converted to 0/1
	// inside the fit.
	C bool
}

// Model is a fitted logistic correctness curve for one cache entry.
//
// The logistic regression is parameterized in the usual machine-learning way,
// P(correct | s) = sigmoid(W*s + B), and then re-expressed in the paper's
// (gamma, t) form:
//
//	gamma = W          // steepness of the curve
//	t     = -B / W     // the similarity at which correctness probability = 0.5
//
// Cov is the 2x2 covariance of (W, B); band.go turns it into a standard error
// on t via the delta method.
type Model struct {
	W, B  float64
	Gamma float64
	T     float64
	Cov   [2][2]float64
	N     int
	// Usable is false when the fit is degenerate (non-positive slope, or a
	// non-finite parameter). Callers treat an unusable model as "we don't know
	// enough" and force exploration, which is always safe.
	Usable bool
}

// sigmoid is the numerically stable logistic function 1 / (1 + e^-x).
// Splitting on the sign of x avoids computing exp of a large positive number,
// which would overflow to +Inf.
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	// For x < 0, rewrite as e^x / (1 + e^x) so we only ever exp a negative.
	ex := math.Exp(x)
	return ex / (1 + ex)
}

// L is the correctness curve in (s, t, gamma) form: the probability the entry
// is correct at similarity s, given threshold t and steepness gamma.
func L(s, t, gamma float64) float64 {
	return sigmoid(gamma * (s - t))
}

// Fit performs L2-regularized logistic regression of C on S (with a bias term)
// using Newton's method / iteratively reweighted least squares (IRLS).
//
// Why regularized, and why we wrote it by hand:
//
//   - Under "perfectly separable" data — e.g. every low-similarity query was
//     wrong and every high-similarity query was right — the unregularized
//     maximum-likelihood fit pushes gamma to infinity (an infinitely sharp
//     step). That is both numerically unstable and statistically overconfident.
//     A small ridge penalty (lambda/2)*W^2 keeps gamma finite. This is part of
//     our cold-start design (see NOTES.md / DECISIONS.md): it is strictly more
//     conservative than the paper's bare formula, so it cannot weaken the
//     guarantee.
//
//   - We also add a tiny ridge on the bias B (biasRidge) purely so the
//     objective is strongly convex and the 2x2 Hessian is always invertible.
//     It is small enough not to meaningfully move the fitted threshold t.
//
//   - The build brief requires this fit be implemented from scratch (no gonum,
//     no ML library). It is only a few dozen lines and it is the code an
//     interviewer will probe, so owning it is the point.
//
// lambda is the ridge strength on the slope W. A small value (e.g. 1e-2) is
// enough to prevent divergence without distorting a well-identified fit.
func Fit(obs []Observation, lambda float64) Model {
	const (
		maxIter   = 50   // Newton converges quadratically; 50 is plenty.
		tol       = 1e-9 // stop when the parameter step is tiny.
		biasRidge = 1e-6 // strong-convexity floor on B (see above).
	)

	m := Model{N: len(obs)}
	if len(obs) == 0 {
		return m // Usable stays false.
	}

	// Parameters theta = (w, b). Start at zero: sigmoid(0)=0.5, a neutral prior.
	w, b := 0.0, 0.0

	// Newton/IRLS loop. Each step solves the 2x2 linear system H * step = grad,
	// where H is the Hessian of the penalized negative log-likelihood and grad
	// is its gradient, then updates theta -= step.
	var h00, h01, h11 float64
	for iter := 0; iter < maxIter; iter++ {
		// Accumulate gradient and Hessian over all observations.
		// Design row for point i is x_i = [s_i, 1].
		//   p_i    = sigmoid(w*s_i + b)          predicted correctness prob
		//   grad  += -(y_i - p_i) * x_i           (from the NLL)
		//   H     +=  p_i(1-p_i) * x_i x_i^T       (Fisher weight)
		var g0, g1 float64 // gradient components (for w, b)
		h00, h01, h11 = 0, 0, 0
		for _, o := range obs {
			y := 0.0
			if o.C {
				y = 1.0
			}
			p := sigmoid(w*o.S + b)
			r := y - p        // residual
			g0 += -r * o.S    // d(NLL)/dw
			g1 += -r          // d(NLL)/db
			wt := p * (1 - p) // Fisher weight
			h00 += wt * o.S * o.S
			h01 += wt * o.S
			h11 += wt
		}
		// Add the ridge penalty's contribution to gradient and Hessian.
		g0 += lambda * w
		g1 += biasRidge * b
		h00 += lambda
		h11 += biasRidge

		// Solve the 2x2 system H * step = g by explicit inverse.
		det := h00*h11 - h01*h01
		if math.Abs(det) < 1e-18 {
			break // singular Hessian; keep the current estimate.
		}
		// step = H^-1 * g
		s0 := (h11*g0 - h01*g1) / det
		s1 := (-h01*g0 + h00*g1) / det
		w -= s0
		b -= s1

		if math.Abs(s0)+math.Abs(s1) < tol {
			break
		}
	}

	// Recompute the final Hessian's inverse as the parameter covariance.
	// (h00,h01,h11 hold the last iteration's Hessian, which is evaluated at the
	// converged theta.) This penalized-Hessian inverse is our estimate of the
	// covariance of (w, b); band.go uses it for the standard error on t. Using
	// the penalized Hessian (rather than the bare Fisher information X^T W X) is
	// what keeps the band finite under separable data — documented as an
	// interpretation in NOTES.md.
	det := h00*h11 - h01*h01
	m.W, m.B = w, b
	m.Gamma = w
	if w != 0 {
		m.T = -b / w
	}
	if det > 0 && math.Abs(det) > 1e-18 {
		m.Cov = [2][2]float64{
			{h11 / det, -h01 / det},
			{-h01 / det, h00 / det},
		}
	}

	// A model is usable only if the slope is meaningfully positive (higher
	// similarity really does predict higher correctness) and all derived
	// quantities are finite. A non-positive or vanishing slope means the data
	// does not support a threshold; the caller will force exploration.
	m.Usable = w > 1e-6 &&
		isFinite(m.T) && isFinite(m.Gamma) &&
		isFinite(m.Cov[0][0]) && isFinite(m.Cov[1][1]) && isFinite(m.Cov[0][1])

	return m
}

func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}
