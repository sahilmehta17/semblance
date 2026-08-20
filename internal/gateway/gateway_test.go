package gateway

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/sahilmehta17/semblance/internal/config"
	"github.com/sahilmehta17/semblance/internal/embed"
	"github.com/sahilmehta17/semblance/internal/judge"
	"github.com/sahilmehta17/semblance/internal/policy"
)

// fixedRand is a deterministic policy.Rand for tests: it always returns v, so
// the explore/exploit coin flip is controllable.
type fixedRand float64

func (f fixedRand) Float64() float64 { return float64(f) }

// testLogger is a discarding logger to keep test output quiet.
func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// testConfig returns a Config with sane cache defaults for tests.
func testConfig(backendURL string, keys ...string) *config.Config {
	return &config.Config{
		BackendBaseURL:     backendURL,
		BackendTimeout:     5 * time.Second,
		APIKeys:            keys,
		Delta:              0.05,
		TemperatureCeiling: 0.3,
		CacheCapacity:      1000,
		MaxObservations:    128,
		CacheShards:        4,
		JudgeQueueSize:     64,
		JudgeWorkers:       2,
	}
}

// newTestServer builds a Server with the deterministic fake embedder, in open
// mode (no API keys). backendURL may be "" for tests that never reach it.
func newTestServer(backendURL string) *Server {
	return newTestServerWithKeys(backendURL)
}

// newTestServerWithKeys is like newTestServer but configures API keys (enabling
// /v1 auth). The exploit coin flip is pinned to 0.999 (exploit unless tau≈1).
func newTestServerWithKeys(backendURL string, keys ...string) *Server {
	logger := testLogger()
	cfg := testConfig(backendURL, keys...)
	return New(cfg, logger,
		WithEmbedder(embed.NewFake(256)),
		WithPolicy(policy.NewPolicy(cfg.Delta, fixedRand(0.999))),
		WithLabeler(judge.NewLabeler(judge.NewDefaultJudge(nil), 64, 2, logger)),
	)
}

func TestHealthz(t *testing.T) {
	h := newTestServer("").Handler()

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	if got, want := rec.Body.String(), `{"status":"ok"}`; got != want {
		t.Errorf("body = %q, want %q", got, want)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
}

func TestHealthzWrongMethod(t *testing.T) {
	h := newTestServer("").Handler()

	req := httptest.NewRequest(http.MethodPost, "/healthz", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusMethodNotAllowed)
	}
}

func TestChatCompletionsWrongMethod(t *testing.T) {
	h := newTestServer("").Handler()

	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusMethodNotAllowed)
	}
}
