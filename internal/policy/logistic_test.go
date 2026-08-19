package policy

import (
	"math"
	"math/rand"
	"testing"
)

// TestFitRecoversKnownSigmoid draws labels from a known logistic curve and
// checks that the fit recovers its (t, gamma) reasonably well. With plenty of
// data and only light regularization, the estimates should be close.
func TestFitRecoversKnownSigmoid(t *testing.T) {
	const (
		trueT     = 0.82
		trueGamma = 30.0
		n         = 20000
	)
	rng := rand.New(rand.NewSource(1))

	obs := make([]Observation, n)
	for i := range obs {
		s := 0.5 + 0.5*rng.Float64() // Uniform(0.5, 1.0)
		p := L(s, trueT, trueGamma)
		obs[i] = Observation{S: s, C: rng.Float64() < p}
	}

	m := Fit(obs, 1e-2)
	if !m.Usable {
		t.Fatalf("fit not usable: %+v", m)
	}
	if math.Abs(m.T-trueT) > 0.02 {
		t.Errorf("t = %.4f, want ~%.4f", m.T, trueT)
	}
	// gamma is harder to pin down; allow a generous band.
	if math.Abs(m.Gamma-trueGamma) > 8 {
		t.Errorf("gamma = %.2f, want ~%.2f", m.Gamma, trueGamma)
	}
	t.Logf("recovered t=%.4f gamma=%.2f seT=%.5f", m.T, m.Gamma, m.seT())
}

// TestFitStaysFiniteOnSeparableData is the regularization safety check: with
// perfectly separable data the unregularized MLE would send gamma to infinity.
// The ridge penalty must keep it finite.
func TestFitStaysFiniteOnSeparableData(t *testing.T) {
	var obs []Observation
	for i := 0; i < 100; i++ {
		s := 0.5 + 0.005*float64(i)                       // 0.50 .. ~1.0
		obs = append(obs, Observation{S: s, C: s > 0.75}) // hard step at 0.75
	}
	m := Fit(obs, 1e-2)
	if !isFinite(m.Gamma) || !isFinite(m.T) {
		t.Fatalf("fit diverged: gamma=%v t=%v", m.Gamma, m.T)
	}
	if m.Gamma <= 0 {
		t.Errorf("expected positive slope, got gamma=%v", m.Gamma)
	}
	t.Logf("separable fit stayed finite: t=%.4f gamma=%.2f", m.T, m.Gamma)
}

// TestFitDegenerateAllSameLabel: if every observation is correct, the fit must
// not claim false certainty — it should either be flagged unusable or carry a
// large standard error so the decision layer stays cautious.
func TestFitDegenerateAllSameLabel(t *testing.T) {
	var obs []Observation
	for i := 0; i < 20; i++ {
		obs = append(obs, Observation{S: 0.7 + 0.01*float64(i), C: true})
	}
	m := Fit(obs, 1e-2)
	// Not asserting Usable either way; asserting the band reflects uncertainty.
	if m.Usable && m.seT() < 0.05 {
		t.Errorf("all-correct data produced overconfident band: seT=%.5f", m.seT())
	}
	t.Logf("all-correct: usable=%v gamma=%.3f t=%.3f seT=%.4f", m.Usable, m.Gamma, m.T, m.seT())
}
