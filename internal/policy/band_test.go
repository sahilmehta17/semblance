package policy

import (
	"math"
	"testing"
)

// TestNormalQuantile checks the probit against well-known reference values.
func TestNormalQuantile(t *testing.T) {
	cases := []struct {
		p, want, tol float64
	}{
		{0.5, 0.0, 1e-6},
		{0.975, 1.959963985, 1e-5}, // classic 95% two-sided z
		{0.995, 2.575829304, 1e-5},
		{0.8413447461, 1.0, 1e-4}, // Phi(1) ≈ 0.8413
		{0.1586552539, -1.0, 1e-4},
	}
	for _, c := range cases {
		got := normalQuantile(c.p)
		if math.Abs(got-c.want) > c.tol {
			t.Errorf("normalQuantile(%g) = %g, want %g (tol %g)", c.p, got, c.want, c.tol)
		}
	}
}

// TestNormalQuantileSymmetry: z(p) = -z(1-p).
func TestNormalQuantileSymmetry(t *testing.T) {
	for _, p := range []float64{0.01, 0.2, 0.37, 0.6, 0.88, 0.99} {
		if a, b := normalQuantile(p), -normalQuantile(1-p); math.Abs(a-b) > 1e-6 {
			t.Errorf("symmetry broken at p=%g: z(p)=%g, -z(1-p)=%g", p, a, b)
		}
	}
}
