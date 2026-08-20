package embed

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

// cosine is a test helper (the cache has its own normalized dot-product path).
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func TestFakeDeterministic(t *testing.T) {
	f := NewFake(256)
	a, _ := f.Embed(context.Background(), "the capital of france")
	b, _ := f.Embed(context.Background(), "the capital of france")
	if cosine(a, b) != 1.0 {
		t.Errorf("same text should embed identically, cosine = %v", cosine(a, b))
	}
}

// TestFakeSimilarityRelationships pins down the properties tests rely on:
// reordering is identical, overlap is high, disjoint is ~0.
func TestFakeSimilarityRelationships(t *testing.T) {
	f := NewFake(512)
	emb := func(s string) []float32 {
		v, _ := f.Embed(context.Background(), s)
		return v
	}

	base := emb("please tell me the capital of france")
	reordered := emb("the capital of france please tell me") // same tokens, new order
	overlap := emb("tell me the capital of germany")         // shares most tokens
	disjoint := emb("completely unrelated words here now")   // no shared tokens

	if c := cosine(base, reordered); math.Abs(c-1.0) > 1e-9 {
		t.Errorf("reordered paraphrase cosine = %v, want 1.0", c)
	}
	if c := cosine(base, overlap); c <= 0.4 || c >= 1.0 {
		t.Errorf("overlap cosine = %v, want strictly between 0.4 and 1.0", c)
	}
	if c := cosine(base, disjoint); math.Abs(c) > 0.2 {
		t.Errorf("disjoint cosine = %v, want near 0", c)
	}
}

// TestOpenAIEmbedderWireProtocol exercises the OpenAI embedder against a fake
// HTTP server: it must send model+input and Bearer auth, and parse the vector.
func TestOpenAIEmbedderWireProtocol(t *testing.T) {
	var gotAuth, gotModel, gotInput string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		var req embeddingRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotModel, gotInput = req.Model, req.Input
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(embeddingResponse{
			Data: []struct {
				Embedding []float32 `json:"embedding"`
			}{{Embedding: []float32{0.1, 0.2, 0.3}}},
		})
	}))
	defer srv.Close()

	e := NewOpenAI(srv.URL+"/v1", "test-key", "text-embedding-3-small", 3, srv.Client())
	vec, err := e.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("got %d dims, want 3", len(vec))
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if gotModel != "text-embedding-3-small" || gotInput != "hello world" {
		t.Errorf("sent model=%q input=%q", gotModel, gotInput)
	}
}

func TestOpenAIEmbedderErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key"}}`))
	}))
	defer srv.Close()

	e := NewOpenAI(srv.URL+"/v1", "nope", "text-embedding-3-small", 1536, srv.Client())
	if _, err := e.Embed(context.Background(), "x"); err == nil {
		t.Error("expected error on non-200 status, got nil")
	}
}
