// Package budget tracks per-API-key spend against a ceiling. When a key is over
// budget the gateway rejects further requests with HTTP 429.
package budget

import "sync"

// Tracker accumulates spend (in cents) per key against a shared ceiling. A
// ceiling of 0 means unlimited (budgets disabled). It is safe for concurrent
// use.
type Tracker struct {
	ceilingCents float64
	mu           sync.Mutex
	spent        map[string]float64 // key -> cents spent
}

// New returns a Tracker with the given per-key ceiling in cents (0 = unlimited).
func New(ceilingCents float64) *Tracker {
	return &Tracker{ceilingCents: ceilingCents, spent: map[string]float64{}}
}

// Enabled reports whether budgets are enforced.
func (t *Tracker) Enabled() bool { return t.ceilingCents > 0 }

// OverBudget reports whether the key has reached or exceeded its ceiling. Always
// false when budgets are disabled.
func (t *Tracker) OverBudget(key string) bool {
	if !t.Enabled() {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.spent[key] >= t.ceilingCents
}

// Add records additional spend (in cents) for a key.
func (t *Tracker) Add(key string, cents float64) {
	if !t.Enabled() || cents == 0 {
		return
	}
	t.mu.Lock()
	t.spent[key] += cents
	t.mu.Unlock()
}

// RemainingCents returns the key's remaining budget, clamped at 0. Returns +Inf
// when budgets are disabled (callers can special-case that if they wish).
func (t *Tracker) RemainingCents(key string) float64 {
	if !t.Enabled() {
		return 0 // caller decides how to represent "unlimited"; gauge stays 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	r := t.ceilingCents - t.spent[key]
	if r < 0 {
		return 0
	}
	return r
}

// Keys returns the keys seen so far (for exporting per-key gauges).
func (t *Tracker) Keys() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	keys := make([]string, 0, len(t.spent))
	for k := range t.spent {
		keys = append(keys, k)
	}
	return keys
}
