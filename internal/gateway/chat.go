package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"

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
		s.logAndWriteError(w, r, http.StatusInternalServerError, "internal_error",
			"failed to build upstream request", err)
		return
	}
	upReq.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(upReq)
	if err != nil {
		// Transport-level failure: backend down, DNS, timeout, client hangup.
		// 502 Bad Gateway is the honest status; detail goes to the log only.
		s.logAndWriteError(w, r, http.StatusBadGateway, "upstream_error",
			"failed to reach the model backend", err)
		return
	}
	defer resp.Body.Close()

	// Decide how to relay. A streaming (SSE) response must be flushed chunk by
	// chunk; a normal JSON response is copied in one shot. The backend's
	// Content-Type is the authoritative signal — text/event-stream — and we also
	// honor the client's stream:true as a hint.
	streaming := req.Stream || isEventStream(resp.Header.Get("Content-Type"))

	s.reqLogger(r).Info("chat completion proxied",
		"model", req.Model, "upstream_status", resp.StatusCode, "streaming", streaming)

	// Propagate the backend's status and Content-Type. A non-2xx from an
	// OpenAI-compatible backend already carries an OpenAI-shaped error body, so
	// relaying it as-is is exactly what a client expects.
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.WriteHeader(resp.StatusCode)

	if streaming {
		s.relayStream(s.reqLogger(r), w, resp.Body)
		return
	}
	if _, err := io.Copy(w, resp.Body); err != nil {
		// The status/headers are already sent, so we can't switch to an error
		// response; just record that the copy was cut short.
		s.reqLogger(r).Warn("response copy interrupted", "err", err)
	}
}

// isEventStream reports whether a Content-Type header denotes Server-Sent
// Events. The header may carry parameters (e.g. "text/event-stream; charset=utf-8"),
// so we match on the media type substring.
func isEventStream(contentType string) bool {
	return strings.Contains(strings.ToLower(contentType), "text/event-stream")
}

// relayStream copies the upstream body to the client chunk by chunk, flushing
// after every write so tokens reach the client as they are produced rather than
// being buffered until the handler returns.
//
// Two idioms here that a Node/TS developer may not have met:
//
//   - net/http buffers writes; bytes are not actually sent to the client until
//     the buffer fills or the handler returns. http.Flusher.Flush() forces them
//     out immediately, which is what makes SSE feel live. We type-assert the
//     ResponseWriter to http.Flusher (the stdlib server's writer implements it);
//     if a wrapper does not, we still deliver correctly — just not incrementally.
//
//   - We read into a fixed buffer and write each piece in a loop instead of
//     io.Copy, because io.Copy gives us no seam to Flush between writes.
//
// Cancellation is not handled explicitly here: the upstream request was built
// with a context derived from the client's request, so a client disconnect
// cancels that context, which makes the next body.Read return an error and ends
// the loop. That is how a client hangup stops the upstream call.
//
// We flush via http.ResponseController rather than a direct w.(http.Flusher)
// assertion, because w is wrapped by our middleware's responseRecorder; the
// controller walks the Unwrap() chain to reach the real writer's Flush().
func (s *Server) relayStream(log *slog.Logger, w http.ResponseWriter, body io.Reader) {
	rc := http.NewResponseController(w)
	// Push the status line and headers out now, so the client opens the stream
	// instead of waiting for the first body bytes. If flushing is unsupported
	// (some writer with no Flusher in its chain), we still relay correctly —
	// just buffered — and note it once.
	flushSupported := rc.Flush() == nil
	if !flushSupported {
		log.Debug("streaming without flush support; delivery may be buffered")
	}

	buf := make([]byte, 4096)
	for {
		n, rerr := body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				// Writing to the client failed — it went away mid-stream. Stop.
				log.Debug("client write failed during stream", "err", werr)
				return
			}
			if flushSupported {
				_ = rc.Flush()
			}
		}
		if rerr != nil {
			// io.EOF is the normal end of stream. context.Canceled means the
			// client disconnected and we cancelled the upstream — also expected.
			if rerr != io.EOF && !errors.Is(rerr, context.Canceled) {
				log.Debug("upstream stream ended", "err", rerr)
			}
			return
		}
	}
}
