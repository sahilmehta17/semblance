package gateway

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// sseBackend is a fake model server that emits n SSE chunks with a delay
// between them, flushing each so its own client (the gateway) receives them
// incrementally.
func sseBackend(t *testing.T, n int, delay time.Duration) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Errorf("backend ResponseWriter is not a Flusher")
			return
		}
		flusher.Flush()
		for i := 0; i < n; i++ {
			fmt.Fprintf(w, "data: chunk %d\n\n", i)
			flusher.Flush()
			time.Sleep(delay)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestStreamingIncrementalDelivery proves tokens are relayed as they arrive, not
// buffered into one block at the end. We time the arrival of each SSE chunk on
// the client side: if the relay were buffered, every chunk would land together
// near the end; streamed, they arrive spread out across the injected delays.
func TestStreamingIncrementalDelivery(t *testing.T) {
	const (
		nChunks = 5
		delay   = 60 * time.Millisecond
	)
	backend := sseBackend(t, nChunks, delay)

	gw := newTestServer(backend.URL + "/v1")
	gw.cfg.BackendTimeout = 30 * time.Second // must not interfere with the stream
	ts := httptest.NewServer(gw.Handler())
	defer ts.Close()

	reqBody := `{"model":"x","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); !isEventStream(ct) {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	// Timestamp each "data:" line as it arrives.
	start := time.Now()
	var arrivals []time.Duration
	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if strings.HasPrefix(line, "data:") {
			arrivals = append(arrivals, time.Since(start))
		}
		if err != nil {
			break // EOF once the backend finishes the stream
		}
	}

	if len(arrivals) != nChunks {
		t.Fatalf("received %d chunks, want %d", len(arrivals), nChunks)
	}

	// The decisive check: the spread between the first and last chunk must be a
	// meaningful fraction of the total injected delay. Buffered delivery would
	// make this near zero (all at the end).
	spread := arrivals[nChunks-1] - arrivals[0]
	minSpread := time.Duration(nChunks-1) * delay / 2
	if spread < minSpread {
		t.Errorf("chunks arrived clustered (spread %v < %v) — delivery looks buffered, not streamed.\narrivals: %v",
			spread, minSpread, arrivals)
	}
	t.Logf("first chunk at %v, last at %v, spread %v", arrivals[0], arrivals[nChunks-1], spread)
}

// TestStreamingClientCancelStopsUpstream proves a client disconnect propagates
// into cancellation of the upstream request. The fake backend sends one chunk
// then blocks on its request context; when the client cancels, the gateway must
// cancel the upstream call, which fires the backend's context and closes the
// `cancelled` channel promptly.
func TestStreamingClientCancelStopsUpstream(t *testing.T) {
	cancelled := make(chan struct{})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: start\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		// Wait until the request context is cancelled. That happens when the
		// gateway aborts this upstream call after its own client disconnects.
		<-r.Context().Done()
		close(cancelled)
	}))
	defer backend.Close()

	gw := newTestServer(backend.URL + "/v1")
	gw.cfg.BackendTimeout = 30 * time.Second
	ts := httptest.NewServer(gw.Handler())
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	reqBody := `{"model":"x","stream":true,"messages":[]}`
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	// Read the first chunk so we know the stream is flowing before we cancel.
	line, _ := bufio.NewReader(resp.Body).ReadString('\n')
	if !strings.HasPrefix(line, "data:") {
		t.Fatalf("expected first SSE chunk, got %q", line)
	}

	// Simulate the client hanging up.
	cancel()

	select {
	case <-cancelled:
		// Upstream saw the cancellation — success.
	case <-time.After(3 * time.Second):
		t.Fatal("upstream was NOT cancelled promptly after client disconnect")
	}
}
