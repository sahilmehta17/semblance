package gateway

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sahilmehta17/semblance/internal/config"
)

// TestRecoverReturns500 verifies a panicking handler yields a 500 OpenAI
// envelope and does not crash the process (the test continuing is the proof).
func TestRecoverReturns500(t *testing.T) {
	s := newTestServer("")
	panicky := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("boom")
	})
	h := s.recoverMiddleware(panicky)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/x", nil))

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	var env errorEnvelope
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("body not an error envelope: %v (%s)", err, rec.Body.String())
	}
	if env.Error.Type != "internal_error" {
		t.Errorf("error type = %q, want internal_error", env.Error.Type)
	}

	// A second request must still work — the server did not fall over.
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, httptest.NewRequest(http.MethodGet, "/x", nil))
	if rec2.Code != http.StatusInternalServerError {
		t.Errorf("second request status = %d, want 500", rec2.Code)
	}
}

// TestRecoverAfterHeadersWritten: if the handler already sent a status (e.g.
// mid-stream) before panicking, recover must NOT try to change it to 500.
func TestRecoverAfterHeadersWritten(t *testing.T) {
	s := newTestServer("")
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("partial "))
		panic("boom mid-stream")
	})
	h := s.recoverMiddleware(handler)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/x", nil))

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (already committed, must not become 500)", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "partial") {
		t.Errorf("expected partial body preserved, got %q", rec.Body.String())
	}
}

// TestRequestIDGeneratedAndEchoed: with no incoming header, an ID is generated,
// echoed on the response, and visible to the inner handler via the context.
func TestRequestIDGeneratedAndEchoed(t *testing.T) {
	s := newTestServer("")
	var seenInHandler string
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenInHandler = RequestIDFrom(r.Context())
	})
	h := s.requestIDMiddleware(handler)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/x", nil))

	echoed := rec.Header().Get(headerRequestID)
	if echoed == "" {
		t.Fatal("no X-Request-Id echoed on response")
	}
	if seenInHandler != echoed {
		t.Errorf("context ID %q != echoed header %q", seenInHandler, echoed)
	}
}

// TestRequestIDPreservesIncoming: a client-supplied X-Request-Id is reused.
func TestRequestIDPreservesIncoming(t *testing.T) {
	s := newTestServer("")
	h := s.requestIDMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))

	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	req.Header.Set(headerRequestID, "client-supplied-123")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if got := rec.Header().Get(headerRequestID); got != "client-supplied-123" {
		t.Errorf("X-Request-Id = %q, want the client-supplied value", got)
	}
}

// TestRequestIDAppearsInLogs runs a request through the full handler with a
// buffered JSON logger and asserts the access log line carries the same ID that
// was echoed to the client.
func TestRequestIDAppearsInLogs(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))
	s := New(&config.Config{BackendBaseURL: "", BackendTimeout: 0}, logger)

	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))

	id := rec.Header().Get(headerRequestID)
	if id == "" {
		t.Fatal("no request id echoed")
	}
	logs := buf.String()
	if !strings.Contains(logs, id) {
		t.Errorf("request id %q not found in logs:\n%s", id, logs)
	}
	if !strings.Contains(logs, `"request_id"`) {
		t.Errorf("logs missing request_id field:\n%s", logs)
	}
}

// --- Auth ---

func TestAuthMissingKey(t *testing.T) {
	h := newTestServerWithKeys("http://127.0.0.1:1/v1", "secret-key").Handler()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	var env errorEnvelope
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("body not an error envelope: %v", err)
	}
	if env.Error.Type != "invalid_request_error" {
		t.Errorf("error type = %q, want invalid_request_error", env.Error.Type)
	}
}

func TestAuthWrongKey(t *testing.T) {
	h := newTestServerWithKeys("http://127.0.0.1:1/v1", "secret-key").Handler()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer wrong-key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", rec.Code)
	}
}

// TestAuthValidKeyPasses: a correct key reaches the handler (proven by getting
// a real completion from a fake backend rather than a 401).
func TestAuthValidKeyPasses(t *testing.T) {
	fb := newFakeBackend(t, http.StatusOK, sampleCompletion, "application/json")
	h := newTestServerWithKeys(fb.baseURL(), "secret-key").Handler()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"x","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer secret-key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (valid key should pass auth)", rec.Code)
	}
	if rec.Body.String() != sampleCompletion {
		t.Errorf("unexpected body: %s", rec.Body.String())
	}
}

// TestAuthHealthzUnauthenticated: health must never require a key even when auth
// is enabled.
func TestAuthHealthzUnauthenticated(t *testing.T) {
	h := newTestServerWithKeys("", "secret-key").Handler()

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))

	if rec.Code != http.StatusOK {
		t.Errorf("healthz status = %d, want 200 without a key", rec.Code)
	}
}

// TestAuthOpenModePasses: with no keys configured, /v1 is reachable without a
// key (dev convenience).
func TestAuthOpenModePasses(t *testing.T) {
	fb := newFakeBackend(t, http.StatusOK, sampleCompletion, "application/json")
	h := newTestServer(fb.baseURL()).Handler() // no keys

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"x","messages":[]}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("open-mode status = %d, want 200", rec.Code)
	}
}
