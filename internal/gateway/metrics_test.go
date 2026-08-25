package gateway

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/sahilmehta17/semblance/internal/cache"
	"github.com/sahilmehta17/semblance/internal/embed"
	"github.com/sahilmehta17/semblance/internal/openai"
	"github.com/sahilmehta17/semblance/internal/policy"
	"github.com/sahilmehta17/semblance/internal/pricing"
)

// TestMetricsRequestOutcomes: a cacheable cold-miss increments the explore
// counter; a bypassable request increments the bypass counter.
func TestMetricsRequestOutcomes(t *testing.T) {
	fb := newFakeBackend(t, http.StatusOK, sampleCompletion, "application/json")
	s := newTestServer(fb.baseURL())
	h := s.Handler()

	postChat(h, `{"model":"llama3.2:1b","messages":[{"role":"user","content":"hello there"}]}`)
	if got := testutil.ToFloat64(s.metrics.RequestsTotal.WithLabelValues("explore")); got != 1 {
		t.Errorf("explore count = %v, want 1", got)
	}

	// stream=true → bypass.
	postChat(h, `{"model":"llama3.2:1b","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if got := testutil.ToFloat64(s.metrics.RequestsTotal.WithLabelValues("bypass")); got != 1 {
		t.Errorf("bypass count = %v, want 1", got)
	}
	if got := testutil.ToFloat64(s.metrics.BypassTotal.WithLabelValues("stream")); got != 1 {
		t.Errorf("bypass reason=stream count = %v, want 1", got)
	}
}

// TestMetricsEndpointUnauthenticated: /metrics is reachable without a key even
// when auth is enabled, and returns Prometheus text.
func TestMetricsEndpointUnauthenticated(t *testing.T) {
	h := newTestServerWithKeys("", "secret-key").Handler()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("/metrics status = %d, want 200 (no key)", rec.Code)
	}
	// delta_target is a plain gauge, always emitted (unlike counters with no
	// children), so it is a stable presence check.
	if !strings.Contains(rec.Body.String(), "semblance_policy_delta_target") {
		t.Error("/metrics missing expected collector")
	}
}

// TestBudget429: an over-budget key is rejected with a 429 OpenAI envelope.
func TestBudget429(t *testing.T) {
	cfg := testConfig("http://127.0.0.1:1/v1", "k")
	cfg.BudgetCents = 100
	s := New(cfg, testLogger(), WithEmbedder(embed.NewFake(256)))
	s.budget.Add("k", 150) // push the key over its 100-cent ceiling
	h := s.Handler()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"x","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer k")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", rec.Code)
	}
	var env errorEnvelope
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("not an error envelope: %v", err)
	}
	if env.Error.Type != "insufficient_quota" {
		t.Errorf("error type = %q, want insufficient_quota", env.Error.Type)
	}
}

// TestCostSavedOnHit: a cache hit records the modeled cost of the avoided
// backend call = the cached answer's token usage priced at the request's model.
func TestCostSavedOnHit(t *testing.T) {
	const model = "test-model"
	// Cached completion carries real usage: 100 prompt + 50 completion tokens.
	cachedBody := `{"id":"c","object":"chat.completion","choices":` +
		`[{"index":0,"message":{"role":"assistant","content":"Paris"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150}}`
	prices := &pricing.Table{Models: map[string]pricing.ModelPrice{
		model: {PromptPer1M: 1.0, CompletionPer1M: 2.0},
	}}
	// expected saved = 100/1e6*1.0 + 50/1e6*2.0
	wantSaved := 100.0/1e6*1.0 + 50.0/1e6*2.0

	fake := embed.NewFake(256)
	store := cache.NewMemoryStore(1000, 128, 1, 1)

	reqJSON := `{"model":"` + model + `","messages":[` +
		`{"role":"system","content":"You are helpful"},` +
		`{"role":"user","content":"please tell me the capital of france"}]}`
	var hr openai.ChatCompletionRequest
	_ = json.Unmarshal([]byte(reqJSON), &hr)
	bucket := bucketKeyForRequest(&hr)

	entryVec, _, _ := fake.Embed(context.Background(), "the capital of france please tell me")
	id := store.Insert(bucket, entryVec, cache.StoredResponse{
		Body: []byte(cachedBody), ContentType: "application/json", Content: "Paris",
	})
	for i := 0; i < 15; i++ {
		store.Observe(bucket, id, 0.55, false)
	}
	for i := 0; i < 15; i++ {
		store.Observe(bucket, id, 0.98, true)
	}

	s := New(testConfig("http://127.0.0.1:1/v1"), testLogger(),
		WithEmbedder(fake), WithStore(store),
		WithPolicy(policy.NewPolicy(0.05, fixedRand(0.999))),
		WithPrices(prices),
	)
	h := s.Handler()

	rec := postChat(h, reqJSON)
	if rec.Code != http.StatusOK || rec.Header().Get(headerCache) != cacheHit {
		t.Fatalf("expected a hit; status=%d cache=%q", rec.Code, rec.Header().Get(headerCache))
	}
	if got := testutil.ToFloat64(s.metrics.CostUSD.WithLabelValues("saved")); math.Abs(got-wantSaved) > 1e-12 {
		t.Errorf("cost_usd_total{saved} = %.9f, want %.9f", got, wantSaved)
	}
}
