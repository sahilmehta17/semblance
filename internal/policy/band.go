package policy

import "math"

// seT returns the standard error of the fitted threshold t = -B/W, propagated
// from the covariance of (W, B) by the delta method.
//
// The delta method: if t = f(W, B) and we know the covariance Sigma of the
// estimator (W, B), then to first order
//
//	Var(t) ≈ ∇f^T · Sigma · ∇f
//
// with the gradient of t = -B/W being
//
//	∂t/∂W =  B / W^2
//	∂t/∂B = -1 / W
//
// Writing Sigma = [[σ_ww, σ_wb], [σ_wb, σ_bb]] and expanding the quadratic form
// gives the expression below. This standard error is what band.go contributes
// to the decision: decide.go pushes t upward by multiples of seT to build a
// pessimistic (worst-case) view of where the true threshold might be.
func (m *Model) seT() float64 {
	w, b := m.W, m.B
	if w == 0 {
		return math.Inf(1)
	}
	dtdw := b / (w * w)
	dtdb := -1 / w

	varT := dtdw*dtdw*m.Cov[0][0] +
		2*dtdw*dtdb*m.Cov[0][1] +
		dtdb*dtdb*m.Cov[1][1]

	if varT < 0 { // tiny negatives can appear from floating-point error
		varT = 0
	}
	return math.Sqrt(varT)
}

// normalQuantile is the inverse CDF (quantile function) of the standard normal
// distribution — i.e. it returns z such that Phi(z) = p. decide.go uses it as
// z(1-eps) to turn a confidence level into a number of standard errors.
//
// We need it because Go's standard library ships the normal CDF's cousins
// (math.Erf/Erfc) but not the inverse, and the build brief forbids pulling in a
// stats library for something this small. This is Acklam's rational
// approximation; its absolute error is below ~1.15e-9 across (0,1), far tighter
// than anything the decision rule needs.
func normalQuantile(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}

	// Coefficients for Acklam's approximation.
	a := [6]float64{-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00}
	bb := [5]float64{-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01}
	c := [6]float64{-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00}
	d := [4]float64{7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00}

	const plow = 0.02425
	const phigh = 1 - plow

	switch {
	case p < plow:
		// Lower tail: substitute q = sqrt(-2 ln p).
		q := math.Sqrt(-2 * math.Log(p))
		return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	case p <= phigh:
		// Central region: rational approximation in q = p - 0.5.
		q := p - 0.5
		r := q * q
		return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r + a[5]) * q /
			(((((bb[0]*r+bb[1])*r+bb[2])*r+bb[3])*r+bb[4])*r + 1)
	default:
		// Upper tail: mirror of the lower tail.
		q := math.Sqrt(-2 * math.Log(1-p))
		return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	}
}
