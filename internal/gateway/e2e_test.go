package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/sahilmehta17/semblance/internal/cache"
	"github.com/sahilmehta17/semblance/internal/embed"
	"github.com/sahilmehta17/semblance/internal/openai"
	"github.com/sahilmehta17/semblance/internal/policy"
)

// TestEndToEndGatewayCacheFlow drives the full wired gateway with the fake
// embedder and shows the three checkpoint behaviors:
//  1. a cold miss (empty bucket) is proxied to the backend and marked miss;
//  2. a paraphrase of a warmed-up entry is served as a HIT without touching the
//     backend, with the cache headers set;
//  3. an identical user turn under a DIFFERENT system prompt does NOT hit (it
//     lands in a different bucket and is a miss).
func TestEndToEndGatewayCacheFlow(t *testing.T) {
	// A backend that counts calls and returns a distinctive live answer, so we
	// can tell a cache hit (backend untouched) from a miss.
	var calls atomic.Int32
	const liveBody = `{"id":"live-1","object":"chat.completion","choices":` +
		`[{"index":0,"message":{"role":"assistant","content":"LIVE-ANSWER"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(liveBody))
	}))
	defer backend.Close()

	fake := embed.NewFake(256)
	store := cache.NewMemoryStore(1000, 128, 1, 1) // single shard: deterministic

	// Seed a CONFIDENT entry in the "helpful" bucket so the exploit is
	// deterministic (the go/no-go simulation already covers the statistics; here
	// we test the wiring). Observations: wrong at low similarity, right at high
	// similarity → a well-identified threshold with tau≈0 at similarity 1.0.
	const cachedBody = `{"id":"cached-1","object":"chat.completion","choices":` +
		`[{"index":0,"message":{"role":"assistant","content":"Paris"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	helpfulReq := `{"model":"llama3.2:1b","messages":[` +
		`{"role":"system","content":"You are a helpful assistant"},` +
		`{"role":"user","content":"please tell me the capital of france"}]}`
	var hr openai.ChatCompletionRequest
	_ = json.Unmarshal([]byte(helpfulReq), &hr)
	helpfulBucket := bucketKeyForRequest(&hr)

	// The stored vector is the same tokens reordered → cosine 1.0 with the query.
	entryVec, _, _ := fake.Embed(context.Background(), "the capital of france please tell me")
	id := store.Insert(helpfulBucket, entryVec, cache.StoredResponse{
		Body: []byte(cachedBody), ContentType: "application/json", Content: "Paris",
	})
	for i := 0; i < 15; i++ {
		store.Observe(helpfulBucket, id, 0.55, false) // wrong when only 0.55 similar
	}
	for i := 0; i < 15; i++ {
		store.Observe(helpfulBucket, id, 0.98, true) // right when 0.98 similar
	}

	srv := New(testConfig(backend.URL+"/v1"), testLogger(),
		WithEmbedder(fake),
		WithStore(store),
		WithPolicy(policy.NewPolicy(0.05, fixedRand(0.999))), // u=0.999 → exploit unless tau≈1
	)
	defer srv.Close()
	h := srv.Handler()

	// --- 1) COLD MISS: a bucket nothing has touched (different system prompt). ---
	before := calls.Load()
	coldReq := `{"model":"llama3.2:1b","messages":[` +
		`{"role":"system","content":"You are a poet"},` +
		`{"role":"user","content":"please tell me the capital of france"}]}`
	rec := postChat(h, coldReq)
	if rec.Code != http.StatusOK {
		t.Fatalf("cold miss: status = %d", rec.Code)
	}
	if got := rec.Header().Get(headerCache); got != cacheMiss {
		t.Errorf("cold miss: X-Semblance-Cache = %q, want %q", got, cacheMiss)
	}
	if rec.Body.String() != liveBody {
		t.Errorf("cold miss: body should be the live backend answer")
	}
	if calls.Load() != before+1 {
		t.Errorf("cold miss: expected 1 backend call, got %d", calls.Load()-before)
	}

	// --- 2) HIT: a reordered paraphrase in the warmed-up helpful bucket. ---
	before = calls.Load()
	rec = postChat(h, helpfulReq)
	if rec.Code != http.StatusOK {
		t.Fatalf("hit: status = %d", rec.Code)
	}
	if got := rec.Header().Get(headerCache); got != cacheHit {
		t.Fatalf("hit: X-Semblance-Cache = %q, want %q", got, cacheHit)
	}
	if rec.Body.String() != cachedBody {
		t.Errorf("hit: body should be the CACHED answer, got %s", rec.Body.String())
	}
	if calls.Load() != before {
		t.Errorf("hit: backend must NOT be called, but call count rose by %d", calls.Load()-before)
	}
	// Headers set on the hit.
	if sim, err := strconv.ParseFloat(rec.Header().Get(headerSimilarity), 64); err != nil || sim < 0.99 {
		t.Errorf("hit: X-Semblance-Similarity = %q, want ~1.0", rec.Header().Get(headerSimilarity))
	}
	if rec.Header().Get(headerTau) == "" {
		t.Error("hit: X-Semblance-Tau not set")
	}

	// --- 3) DIFFERENT SYSTEM PROMPT: identical user turn must NOT hit. ---
	before = calls.Load()
	sarcasticReq := `{"model":"llama3.2:1b","messages":[` +
		`{"role":"system","content":"You are a sarcastic assistant"},` +
		`{"role":"user","content":"please tell me the capital of france"}]}`
	rec = postChat(h, sarcasticReq)
	if got := rec.Header().Get(headerCache); got != cacheMiss {
		t.Errorf("different system prompt: X-Semblance-Cache = %q, want %q", got, cacheMiss)
	}
	if rec.Body.String() == cachedBody {
		t.Error("different system prompt must NOT be served the other bucket's cached answer")
	}
	if calls.Load() != before+1 {
		t.Errorf("different system prompt: expected a backend call (miss), got %d", calls.Load()-before)
	}
}
