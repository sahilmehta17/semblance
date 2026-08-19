package gateway

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/sahilmehta17/semblance/internal/config"
)

// newTestServer builds a Server whose backend points at backendURL, in open
// mode (no API keys). A discarding logger keeps test output quiet. backendURL
// may be "" for tests that never reach the backend (e.g. healthz, invalid-JSON).
func newTestServer(backendURL string) *Server {
	return newTestServerWithKeys(backendURL)
}

// newTestServerWithKeys is like newTestServer but configures the given API keys,
// enabling auth on the /v1 routes.
func newTestServerWithKeys(backendURL string, keys ...string) *Server {
	cfg := &config.Config{
		BackendBaseURL: backendURL,
		BackendTimeout: 5 * time.Second,
		APIKeys:        keys,
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return New(cfg, logger)
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
