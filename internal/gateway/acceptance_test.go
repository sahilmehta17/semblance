package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/sahilmehta17/semblance/internal/openai"
)

// TestAcceptanceOllamaChatCompletion is the Step 2 acceptance test from the
// brief: a real request through the gateway to a real Ollama backend returns a
// correct completion. It is DEFERRED until Ollama is installed.
//
// It runs only when SEMBLANCE_OLLAMA_ACCEPTANCE=1 is set (and Ollama is serving
// its OpenAI endpoint at SEMBLANCE_BACKEND_URL, default http://localhost:11434/v1).
// Otherwise it skips with a clear marker so `go test ./...` stays green before
// the backend exists.
//
// NOTE: the brief phrases the acceptance criterion as "an unmodified OpenAI SDK
// client works". Because this project is stdlib-only (no OpenAI Go SDK
// dependency allowed), this Go test exercises the same HTTP contract the SDK
// uses. A tiny Python script driving the official openai package against
// http://localhost:8080/v1 will be added alongside once Ollama is up, to prove
// SDK-level compatibility end to end.
func TestAcceptanceOllamaChatCompletion(t *testing.T) {
	if os.Getenv("SEMBLANCE_OLLAMA_ACCEPTANCE") != "1" {
		t.Skip("DEFERRED: set SEMBLANCE_OLLAMA_ACCEPTANCE=1 with Ollama running to enable")
	}

	backend := os.Getenv("SEMBLANCE_BACKEND_URL")
	if backend == "" {
		backend = "http://localhost:11434/v1"
	}
	model := os.Getenv("SEMBLANCE_ACCEPTANCE_MODEL")
	if model == "" {
		model = "llama3.2:1b"
	}

	// Stand up the real gateway handler pointed at the real backend.
	gw := newTestServer(backend)
	gw.cfg.BackendTimeout = 120 * time.Second
	ts := httptest.NewServer(gw.Handler())
	defer ts.Close()

	reqBody := `{"model":"` + model + `","messages":[` +
		`{"role":"user","content":"Reply with exactly the word: pong"}],` +
		`"temperature":0}`

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, raw)
	}

	var out openai.ChatCompletionResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("response not a chat completion: %v (%s)", err, raw)
	}
	if len(out.Choices) == 0 {
		t.Fatalf("no choices in response: %s", raw)
	}
	if out.Usage.TotalTokens == 0 {
		t.Errorf("expected non-zero usage.total_tokens, got %s", raw)
	}
	t.Logf("ollama replied: %s (tokens: %d)", out.Choices[0].Message.Content, out.Usage.TotalTokens)
}
