package budget

import "testing"

func TestBudgetLifecycle(t *testing.T) {
	tr := New(100) // 100-cent ceiling
	if !tr.Enabled() {
		t.Fatal("should be enabled")
	}
	if tr.OverBudget("k") {
		t.Error("fresh key should not be over budget")
	}
	tr.Add("k", 60)
	if tr.OverBudget("k") {
		t.Error("60 < 100, not over budget")
	}
	if got := tr.RemainingCents("k"); got != 40 {
		t.Errorf("remaining = %v, want 40", got)
	}
	tr.Add("k", 50) // total 110
	if !tr.OverBudget("k") {
		t.Error("110 >= 100, should be over budget")
	}
	if got := tr.RemainingCents("k"); got != 0 {
		t.Errorf("remaining = %v, want 0 (clamped)", got)
	}
	// Keys are independent.
	if tr.OverBudget("other") {
		t.Error("a different key has its own budget")
	}
}

func TestBudgetDisabled(t *testing.T) {
	tr := New(0) // unlimited
	if tr.Enabled() {
		t.Error("ceiling 0 means disabled")
	}
	tr.Add("k", 1e9)
	if tr.OverBudget("k") {
		t.Error("disabled budget is never over")
	}
}
