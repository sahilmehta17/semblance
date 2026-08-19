package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"

	"github.com/sahilmehta17/semblance/internal/openai"
)

// maxRequestBytes caps how much we read from a client to avoid unbounded memory
// use from a hostile or buggy caller. 10 MiB is far above any real chat body.
const maxRequestBytes = 10 << 20

// handleChatCompletions is the Step 2 non-streaming passthrough. It reads the
// client's request, forwards the ORIGINAL bytes to the backend, and copies the
// backend's response back unchanged. Forwarding raw bytes (rather than
// re-encoding a parsed struct) is what guarantees no client parameter is ever
// dropped — see the internal/openai package doc.
//
// Streaming (stream=true) is Step 3; this handler simply relays whatever the
// backend returns without incremental flushing.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	// Read the full body, capped. MaxBytesReader makes an over-large body fail
	// here instead of ballooning memory.
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
		return
	}

	// Validate that the body is JSON at all. We do NOT strictly decode into our
	// struct for validation, because that could reject an unusual-but-valid
	// request the backend would accept; json.Valid is the lenient check.
	if !json.Valid(body) {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "request body is not valid JSON")
		return
	}

	// Best-effort decode purely for logging/inspection. Errors are ignored: the
	// body is already known to be valid JSON, and the forwarded bytes are the
	// original regardless of what we manage to parse.
	var req openai.ChatCompletionRequest
	_ = json.Unmarshal(body, &req)

	// Bound the upstream call with a context derived from the request, so a
	// client disconnect (r.Context() cancelled) or our timeout both abort it.
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.BackendTimeout)
	defer cancel()

	upstreamURL := s.cfg.BackendBaseURL + "/chat/completions"
	upReq, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(body))
	if err != nil {
		s.logAndWriteError(w, http.StatusInternalServerError, "internal_error",
			"failed to build upstream request", err)
		return
	}
	upReq.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(upReq)
	if err != nil {
		// Transport-level failure: backend down, DNS, timeout, client hangup.
		// 502 Bad Gateway is the honest status; detail goes to the log only.
		s.logAndWriteError(w, http.StatusBadGateway, "upstream_error",
			"failed to reach the model backend", err)
		return
	}
	defer resp.Body.Close()

	s.logger.Info("chat completion proxied",
		"model", req.Model, "upstream_status", resp.StatusCode)

	// Pass the response through unchanged: propagate the backend's status and
	// its Content-Type, then stream the body. A non-2xx from an OpenAI-compatible
	// backend already carries an OpenAI-shaped error body, so relaying it as-is
	// is exactly what a client expects.
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		// The status/headers are already sent, so we can't switch to an error
		// response; just record that the copy was cut short.
		s.logger.Warn("response copy interrupted", "err", err)
	}
}
