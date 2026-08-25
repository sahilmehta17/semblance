package metrics

import (
	"io"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestCountersAndGaugesIncrement(t *testing.T) {
	m := New()

	m.RequestsTotal.WithLabelValues("exploit").Inc()
	m.RequestsTotal.WithLabelValues("exploit").Inc()
	m.RequestsTotal.WithLabelValues("explore").Inc()
	if got := testutil.ToFloat64(m.RequestsTotal.WithLabelValues("exploit")); got != 2 {
		t.Errorf("exploit count = %v, want 2", got)
	}
	if got := testutil.ToFloat64(m.RequestsTotal.WithLabelValues("explore")); got != 1 {
		t.Errorf("explore count = %v, want 1", got)
	}

	m.BypassTotal.WithLabelValues("stream").Inc()
	if got := testutil.ToFloat64(m.BypassTotal.WithLabelValues("stream")); got != 1 {
		t.Errorf("bypass stream = %v, want 1", got)
	}

	m.BackendTokens.WithLabelValues("prompt").Add(100)
	m.BackendTokens.WithLabelValues("completion").Add(50)
	if got := testutil.ToFloat64(m.BackendTokens.WithLabelValues("prompt")); got != 100 {
		t.Errorf("prompt tokens = %v, want 100", got)
	}

	m.EmbedTokens.Add(7)
	if got := testutil.ToFloat64(m.EmbedTokens); got != 7 {
		t.Errorf("embed tokens = %v, want 7", got)
	}

	m.CostUSD.WithLabelValues("saved").Add(0.25)
	if got := testutil.ToFloat64(m.CostUSD.WithLabelValues("saved")); got != 0.25 {
		t.Errorf("cost saved = %v, want 0.25", got)
	}

	m.BudgetCents.WithLabelValues("k").Set(42)
	if got := testutil.ToFloat64(m.BudgetCents.WithLabelValues("k")); got != 42 {
		t.Errorf("budget = %v, want 42", got)
	}

	m.DeltaTarget.Set(0.02)
	if got := testutil.ToFloat64(m.DeltaTarget); got != 0.02 {
		t.Errorf("delta target = %v, want 0.02", got)
	}
}

// TestScrapeExposesMetrics checks the /metrics text format and that observed
// histograms show up with the right sample counts.
func TestScrapeExposesMetrics(t *testing.T) {
	m := New()
	m.Tau.Observe(0.3)
	m.Similarity.Observe(0.95)
	m.RefitSeconds.Observe(0.0001)
	m.EntryObs.Observe(8)
	// Counters emit nothing until a child is created, so touch the ones we
	// assert on below.
	m.RequestsTotal.WithLabelValues("explore").Inc()
	m.CostUSD.WithLabelValues("spent").Add(0.1)
	m.BackendTokens.WithLabelValues("prompt").Add(1)

	rec := httptest.NewRecorder()
	m.Handler().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	body, _ := io.ReadAll(rec.Body)
	text := string(body)

	// Prometheus exposition format.
	if !strings.Contains(text, "# HELP semblance_requests_total") {
		t.Error("missing HELP for requests_total")
	}
	for _, want := range []string{
		"semblance_tau_count 1",
		"semblance_similarity_count 1",
		"semblance_policy_refit_seconds_count 1",
		"semblance_entry_observations_count 1",
		"semblance_cost_usd_total",
		"semblance_backend_tokens_total",
		"semblance_policy_delta_target",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("scrape missing %q", want)
		}
	}
}

// TestGaugeFuncRegistration confirms the scrape-time gauges reflect live values.
func TestGaugeFuncRegistration(t *testing.T) {
	m := New()
	n := 3
	m.RegisterCacheEntriesFunc(func() float64 { return float64(n) })

	scrape := func() string {
		rec := httptest.NewRecorder()
		m.Handler().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
		b, _ := io.ReadAll(rec.Body)
		return string(b)
	}
	if !strings.Contains(scrape(), "semblance_cache_entries 3") {
		t.Error("cache_entries gauge func not reflecting value 3")
	}
	n = 10
	if !strings.Contains(scrape(), "semblance_cache_entries 10") {
		t.Error("cache_entries gauge func should re-evaluate on scrape")
	}
}
