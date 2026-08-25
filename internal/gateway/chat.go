package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/sahilmehta17/semblance/internal/cache"
	"github.com/sahilmehta17/semblance/internal/judge"
	"github.com/sahilmehta17/semblance/internal/openai"
)

// maxRequestBytes caps how much we read from a client to avoid unbounded memory
// use from a hostile or buggy caller. 10 MiB is far above any real chat body.
const maxRequestBytes = 10 << 20

// Response headers that expose the cache decision to clients and tests.
const (
	headerCache      = "X-Semblance-Cache"      // hit | miss | bypass
	headerSimilarity = "X-Semblance-Similarity" // cosine to nearest neighbor
	headerTau        = "X-Semblance-Tau"        // exploration probability used
)

const (
	cacheHit    = "hit"
	cacheMiss   = "miss"
	cacheBypass = "bypass"
)

// handleChatCompletions is the verified-cache decision path.
//
// Shape of the logic:
//  1. Validate the body. Forwarding always uses the ORIGINAL bytes, so unknown
//     client fields are never dropped.
//  2. Bypass (no caching) for streaming, tools/functions, n>1, or temperature
//     above the ceiling — or when no embedder is configured.
//  3. Otherwise embed the final user turn, find the nearest neighbor within the
//     exact-match bucket, and ask the policy to EXPLORE or EXPLOIT.
//  4. EXPLOIT serves the cached answer; EXPLORE proxies to the backend, then
//     asynchronously labels equivalence and updates the cache (Algorithm 1).
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
		return
	}
	if !json.Valid(body) {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "request body is not valid JSON")
		return
	}

	// Best-effort decode for inspection; forwarding uses the original bytes.
	var req openai.ChatCompletionRequest
	_ = json.Unmarshal(body, &req)

	key := apiKeyFrom(r.Context())
	model := req.Model

	// Caching entirely disabled (no embedder) → pure passthrough.
	if s.embedder == nil {
		s.countBypass("no_embedder")
		w.Header().Set(headerCache, cacheBypass)
		s.proxyPassthrough(w, r, body, &req)
		return
	}

	// Bypass for request shapes we do not (or cannot) cache.
	if reason := s.bypassReason(&req); reason != "" {
		s.countBypass(reason)
		s.reqLogger(r).Debug("cache bypass", "reason", reason)
		w.Header().Set(headerCache, cacheBypass)
		s.proxyPassthrough(w, r, body, &req)
		return
	}

	finalText, _ := finalUserText(&req) // bypassReason guaranteed this is present
	bucket := bucketKeyForRequest(&req)

	qvec, embTokens, err := s.embedder.Embed(r.Context(), finalText)
	if err != nil {
		// Can't embed → can't cache. Fall back to a plain proxied miss.
		s.reqLogger(r).Warn("embed failed; passthrough without caching", "err", err)
		s.countBypass("embed_error")
		w.Header().Set(headerCache, cacheMiss)
		s.proxyPassthrough(w, r, body, &req)
		return
	}
	// An embedding happens on hits and misses alike (to find the neighbor).
	s.recordEmbedUsage(model, key, embTokens)

	match, found := s.store.Nearest(bucket, qvec)
	if !found {
		s.metrics.RequestsTotal.WithLabelValues("explore").Inc()
		s.exploreColdMiss(w, r, body, bucket, qvec, model, key)
		return
	}

	// Neighbor found: record the observable signals, then let the policy decide.
	// Decide() draws u internally from the injected RNG, so this single call
	// covers "get tau_hat, draw u, explore-or-exploit". We time it as the refit
	// cost (the logistic fit dominates).
	s.metrics.Similarity.Observe(match.Similarity)
	s.metrics.EntryObs.Observe(float64(len(match.Observations)))
	start := time.Now()
	decision := s.policy.Decide(match.Observations, match.Similarity)
	s.metrics.RefitSeconds.Observe(time.Since(start).Seconds())
	s.metrics.Tau.Observe(decision.Tau)

	w.Header().Set(headerSimilarity, formatFloat(match.Similarity))
	w.Header().Set(headerTau, formatFloat(decision.Tau))

	if decision.Explore {
		s.metrics.RequestsTotal.WithLabelValues("explore").Inc()
		s.exploreWithNeighbor(w, r, body, bucket, qvec, match, model, key)
		return
	}
	s.metrics.RequestsTotal.WithLabelValues("exploit").Inc()
	s.serveHit(w, r, match, model)
}

// countBypass records a bypass in both the legacy atomic counter and Prometheus.
func (s *Server) countBypass(reason string) {
	s.bypassTotal.Add(1)
	s.metrics.BypassTotal.WithLabelValues(reason).Inc()
	s.metrics.RequestsTotal.WithLabelValues("bypass").Inc()
}

// serveHit returns the cached response (EXPLOIT). Learning happens only on
// EXPLORE, so we do not record an observation here.
func (s *Server) serveHit(w http.ResponseWriter, r *http.Request, match cache.Match, model string) {
	w.Header().Set(headerCache, cacheHit)
	ct := match.Response.ContentType
	if ct == "" {
		ct = "application/json"
	}
	w.Header().Set("Content-Type", ct)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(match.Response.Body)
	// The modeled cost of the backend call this hit avoided.
	s.recordSavedCost(model, match.Response.Body)
	s.reqLogger(r).Info("cache hit", "similarity", match.Similarity)
}

// exploreColdMiss handles an empty bucket: there is no neighbor, so we must
// call the backend and (on success) insert this prompt as the bucket's first
// entry — unconditionally, since there is nothing to guard against.
func (s *Server) exploreColdMiss(w http.ResponseWriter, r *http.Request, body []byte, bucket string, qvec []float32, model, key string) {
	w.Header().Set(headerCache, cacheMiss)
	w.Header().Set(headerTau, "1") // forced explore (cold start), no neighbor

	res, err := s.fetchUpstreamBuffered(r, body)
	if err != nil {
		s.logAndWriteError(w, r, http.StatusBadGateway, "upstream_error", "failed to reach the model backend", err)
		return
	}
	s.writeBuffered(w, res)

	if res.cacheable() {
		s.recordBackendUsage(model, key, res.body)
		s.store.Insert(bucket, qvec, cache.StoredResponse{
			Body:        res.body,
			ContentType: res.contentType,
			Content:     extractContent(res.body),
		})
	}
}

// exploreWithNeighbor handles a MISS where a neighbor exists: proxy to the
// backend, return the fresh answer, then asynchronously label whether the
// neighbor's answer would have been correct and update the cache per
// Algorithm 1 (append observation; guarded-insert only when the neighbor was
// wrong).
func (s *Server) exploreWithNeighbor(w http.ResponseWriter, r *http.Request, body []byte, bucket string, qvec []float32, match cache.Match, model, key string) {
	w.Header().Set(headerCache, cacheMiss)

	res, err := s.fetchUpstreamBuffered(r, body)
	if err != nil {
		s.logAndWriteError(w, r, http.StatusBadGateway, "upstream_error", "failed to reach the model backend", err)
		return
	}
	s.writeBuffered(w, res)

	if !res.cacheable() {
		return
	}
	s.recordBackendUsage(model, key, res.body)

	newResp := cache.StoredResponse{
		Body:        res.body,
		ContentType: res.contentType,
		Content:     extractContent(res.body),
	}
	// Capture values for the async closure (avoid capturing the request).
	entryID, sim := match.EntryID, match.Similarity
	reference := match.Response.Content

	// Off the critical path: judge equivalence, then record the observation and
	// guarded-insert. Dropped if the queue is full (safe: we just learn less).
	// The label also feeds the honest observable proxy: judge-observed c=0 rate.
	s.labeler.Submit(judge.Job{
		Reference: reference,
		Candidate: newResp.Content,
		OnResult: func(equivalent bool) {
			s.metrics.ExploresLabel.Inc()
			if !equivalent {
				s.metrics.ExploresC0.Inc()
			}
			s.store.Observe(bucket, entryID, sim, equivalent)
			cache.GuardedInsert(s.store, bucket, qvec, newResp, equivalent)
		},
	})
}

// bypassReason returns why a request cannot be cached, or "" if it is
// cacheable. High-temperature, streaming, multi-choice, and tool-using requests
// are not safely cacheable.
func (s *Server) bypassReason(req *openai.ChatCompletionRequest) string {
	switch {
	case req.Stream:
		return "stream"
	case len(req.Tools) > 0:
		return "tools"
	case len(req.Functions) > 0:
		return "functions"
	case req.N != nil && *req.N > 1:
		return "n>1"
	case req.Temperature != nil && *req.Temperature > s.cfg.TemperatureCeiling:
		return "temperature"
	}
	if _, ok := finalUserText(req); !ok {
		return "no_user_turn"
	}
	return ""
}

// --- upstream helpers ---

// upstreamResult is a buffered (non-streaming) backend response.
type upstreamResult struct {
	status      int
	contentType string
	body        []byte
}

// cacheable reports whether this response should be stored/learned from: a
// successful, non-empty completion.
func (u *upstreamResult) cacheable() bool {
	return u.status >= 200 && u.status < 300 && len(u.body) > 0
}

// fetchUpstreamBuffered performs the backend call and reads the full response.
// Used on the cache paths (which are always non-streaming — streaming bypasses).
func (s *Server) fetchUpstreamBuffered(r *http.Request, body []byte) (*upstreamResult, error) {
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.BackendTimeout)
	defer cancel()

	upReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BackendBaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	upReq.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(upReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return &upstreamResult{status: resp.StatusCode, contentType: resp.Header.Get("Content-Type"), body: b}, nil
}

// writeBuffered relays a buffered upstream response to the client.
func (s *Server) writeBuffered(w http.ResponseWriter, res *upstreamResult) {
	if res.contentType != "" {
		w.Header().Set("Content-Type", res.contentType)
	}
	w.WriteHeader(res.status)
	_, _ = w.Write(res.body)
}

// proxyPassthrough forwards the original request bytes and relays the response,
// streaming (SSE) chunk-by-chunk when appropriate. Used for bypassed requests.
func (s *Server) proxyPassthrough(w http.ResponseWriter, r *http.Request, body []byte, req *openai.ChatCompletionRequest) {
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.BackendTimeout)
	defer cancel()

	upReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BackendBaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		s.logAndWriteError(w, r, http.StatusInternalServerError, "internal_error", "failed to build upstream request", err)
		return
	}
	upReq.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(upReq)
	if err != nil {
		s.logAndWriteError(w, r, http.StatusBadGateway, "upstream_error", "failed to reach the model backend", err)
		return
	}
	defer resp.Body.Close()

	streaming := req.Stream || isEventStream(resp.Header.Get("Content-Type"))
	s.reqLogger(r).Info("chat completion proxied",
		"model", req.Model, "upstream_status", resp.StatusCode, "streaming", streaming)

	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.WriteHeader(resp.StatusCode)

	if streaming {
		s.relayStream(s.reqLogger(r), w, resp.Body)
		return
	}
	if _, err := io.Copy(w, resp.Body); err != nil {
		s.reqLogger(r).Warn("response copy interrupted", "err", err)
	}
}

// isEventStream reports whether a Content-Type header denotes Server-Sent Events.
func isEventStream(contentType string) bool {
	return strings.Contains(strings.ToLower(contentType), "text/event-stream")
}

// relayStream copies the upstream body to the client chunk by chunk, flushing
// after every write so tokens reach the client as they are produced.
//
// We flush via http.ResponseController rather than a direct w.(http.Flusher)
// assertion, because w is wrapped by the middleware's responseRecorder; the
// controller walks the Unwrap() chain to reach the real writer's Flush().
// Cancellation is implicit: the upstream context derives from the client's
// request, so a client disconnect cancels it and the next Read returns an error.
func (s *Server) relayStream(log *slog.Logger, w http.ResponseWriter, body io.Reader) {
	rc := http.NewResponseController(w)
	flushSupported := rc.Flush() == nil
	if !flushSupported {
		log.Debug("streaming without flush support; delivery may be buffered")
	}

	buf := make([]byte, 4096)
	for {
		n, rerr := body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				log.Debug("client write failed during stream", "err", werr)
				return
			}
			if flushSupported {
				_ = rc.Flush()
			}
		}
		if rerr != nil {
			if rerr != io.EOF && !errors.Is(rerr, context.Canceled) {
				log.Debug("upstream stream ended", "err", rerr)
			}
			return
		}
	}
}

// --- request/response helpers ---

// bucketKeyForRequest builds the exact-match bucket key. Everything that must
// isolate cache entries goes in: model, system prompt, sampling params, and all
// prior turns. Only the FINAL user turn is matched semantically (via embedding),
// so it is deliberately excluded from the key.
func bucketKeyForRequest(req *openai.ChatCompletionRequest) string {
	parts := []string{
		req.Model,
		systemPrompt(req),
		floatPtrString(req.Temperature),
		floatPtrString(req.TopP),
	}
	// Prior messages = all but the final turn.
	for i := 0; i < len(req.Messages)-1; i++ {
		m := req.Messages[i]
		txt, _ := messageText(m.Content)
		parts = append(parts, m.Role, txt)
	}
	return cache.BucketKey(parts...)
}

// systemPrompt concatenates the text of all system messages.
func systemPrompt(req *openai.ChatCompletionRequest) string {
	var b strings.Builder
	for _, m := range req.Messages {
		if m.Role == "system" {
			if txt, ok := messageText(m.Content); ok {
				b.WriteString(txt)
				b.WriteByte('\n')
			}
		}
	}
	return b.String()
}

// finalUserText returns the text of the final turn, ok only if it is a user
// turn with plain-string content (what we can embed).
func finalUserText(req *openai.ChatCompletionRequest) (string, bool) {
	if len(req.Messages) == 0 {
		return "", false
	}
	last := req.Messages[len(req.Messages)-1]
	if last.Role != "user" {
		return "", false
	}
	return messageText(last.Content)
}

// messageText decodes message content. OpenAI content may be a plain string or
// an array of typed parts; we return (string, true) only for the plain-string
// form, and (rawJSON, false) otherwise so callers can decide.
func messageText(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, true
	}
	return string(raw), false
}

// extractContent pulls the assistant's answer text out of a completion response
// body, for equivalence labeling.
func extractContent(body []byte) string {
	var resp openai.ChatCompletionResponse
	if err := json.Unmarshal(body, &resp); err != nil || len(resp.Choices) == 0 {
		return ""
	}
	if txt, ok := messageText(resp.Choices[0].Message.Content); ok {
		return txt
	}
	return string(resp.Choices[0].Message.Content)
}

func floatPtrString(f *float64) string {
	if f == nil {
		return ""
	}
	return formatFloat(*f)
}

func formatFloat(f float64) string {
	return strconv.FormatFloat(f, 'f', 4, 64)
}
