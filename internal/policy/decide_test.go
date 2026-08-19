package policy

import (
	"math"
	"math/rand"
	"testing"
)

// constRand is a deterministic Rand that always returns the same value. Handy
// for pinning down the explore/exploit coin flip in a test.
type constRand float64

func (c constRand) Float64() float64 { return float64(c) }

func TestColdStartForcesExplore(t *testing.T) {
	p := NewPolicy(0.02, constRand(0.0))
	// Fewer than NMin (=5) observations: must explore regardless of similarity.
	obs := []Observation{{S: 0.99, C: true}, {S: 0.98, C: true}}
	d := p.Decide(obs, 0.99)
	if !d.Explore || !d.Forced {
		t.Errorf("cold start: got Explore=%v Forced=%v, want both true", d.Explore, d.Forced)
	}
	if d.Tau != 1 {
		t.Errorf("cold start tau = %v, want 1", d.Tau)
	}
}

// buildObs makes a clean, well-separated observation set around threshold t.
func buildObs(threshold float64) []Observation {
	rng := rand.New(rand.NewSource(7))
	var obs []Observation
	for i := 0; i < 400; i++ {
		s := 0.5 + 0.5*rng.Float64()
		p := L(s, threshold, 40)
		obs = append(obs, Observation{S: s, C: rng.Float64() < p})
	}
	return obs
}

// TestTauMonotoneInSimilarity: with a fixed fitted model, tau should not
// increase as similarity increases — higher similarity means more confidence
// to exploit, hence less exploration.
func TestTauMonotoneInSimilarity(t *testing.T) {
	p := NewPolicy(0.02, constRand(0.5))
	obs := buildObs(0.80)
	m := Fit(obs, p.Lambda)
	if !m.Usable {
		t.Fatal("model not usable")
	}
	prev := math.Inf(1)
	for s := 0.60; s <= 1.0001; s += 0.02 {
		tau := p.tauHat(&m, s)
		if tau > prev+1e-9 {
			t.Errorf("tau increased with similarity: s=%.2f tau=%.4f > prev=%.4f", s, tau, prev)
		}
		prev = tau
	}
}

// TestExploitAtHighSimilarity: well above the threshold, tau should be small
// enough that a coin flip near 1.0 exploits.
func TestExploitAtHighSimilarity(t *testing.T) {
	// u = 0.999: only explores if tau ~ 1. At high similarity we expect exploit.
	p := NewPolicy(0.05, constRand(0.999))
	obs := buildObs(0.80)
	d := p.Decide(obs, 0.99)
	if d.Explore {
		t.Errorf("expected EXPLOIT at s=0.99 well above threshold, got explore (tau=%.4f)", d.Tau)
	}
	t.Logf("s=0.99 tau=%.4f -> exploit", d.Tau)
}

// TestExploreAtLowSimilarity: near/below the threshold the entry is unreliable,
// so tau should be high and a mid coin flip explores.
func TestExploreAtLowSimilarity(t *testing.T) {
	p := NewPolicy(0.02, constRand(0.5))
	obs := buildObs(0.85)
	d := p.Decide(obs, 0.70) // below threshold
	if !d.Explore {
		t.Errorf("expected EXPLORE at s=0.70 below threshold, got exploit (tau=%.4f)", d.Tau)
	}
	t.Logf("s=0.70 tau=%.4f -> explore", d.Tau)
}
