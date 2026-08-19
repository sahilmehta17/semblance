package gateway

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"log/slog"
	"net/http"
	"runtime/debug"
	"strings"
	"time"
)

// headerRequestID is the request-correlation header we read and echo.
const headerRequestID = "X-Request-Id"

// ctxKey is an unexported type for context keys. Using a private type (rather
// than a bare string) guarantees our keys can never collide with keys set by
// other packages sharing the same context — the standard Go idiom.
type ctxKey int

const (
	ctxKeyRequestID ctxKey = iota
	ctxKeyLogger
)

// RequestIDFrom returns the request ID stored on the context, or "" if none.
func RequestIDFrom(ctx context.Context) string {
	id, _ := ctx.Value(ctxKeyRequestID).(string)
	return id
}

// loggerFrom returns the request-scoped logger on the context, or nil if none.
func loggerFrom(ctx context.Context) *slog.Logger {
	lg, _ := ctx.Value(ctxKeyLogger).(*slog.Logger)
	return lg
}

// reqLogger returns the logger carrying this request's ID, falling back to the
// server logger when the request never passed through requestIDMiddleware
// (e.g. in a narrow unit test).
func (s *Server) reqLogger(r *http.Request) *slog.Logger {
	if lg := loggerFrom(r.Context()); lg != nil {
		return lg
	}
	return s.logger
}

// responseRecorder wraps an http.ResponseWriter to remember the status code and
// whether anything has been written yet. recover uses wroteHeader to decide if
// it can still send a 500; access logging uses status.
//
// It embeds http.ResponseWriter so Header()/Write()/WriteHeader() are inherited
// and only overridden where we need to observe them. Crucially it also exposes
// Unwrap(), so http.ResponseController can still reach the underlying writer's
// Flush() through this wrapper (see relayStream) — without Unwrap, wrapping the
// writer would silently disable streaming flushes.
type responseRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (rr *responseRecorder) WriteHeader(code int) {
	if !rr.wroteHeader {
		rr.status = code
		rr.wroteHeader = true
	}
	rr.ResponseWriter.WriteHeader(code)
}

func (rr *responseRecorder) Write(b []byte) (int, error) {
	if !rr.wroteHeader {
		// A Write without an explicit WriteHeader implies 200 (net/http does
		// this); record it so our status is accurate.
		rr.status = http.StatusOK
		rr.wroteHeader = true
	}
	return rr.ResponseWriter.Write(b)
}

// Unwrap lets http.ResponseController find the underlying writer's optional
// interfaces (Flusher, etc.).
func (rr *responseRecorder) Unwrap() http.ResponseWriter { return rr.ResponseWriter }

// recoverMiddleware is the OUTERMOST layer. It catches any panic from an inner
// handler, logs it with a stack trace, and — if the response has not started —
// returns a 500 in the OpenAI error envelope. If bytes were already sent (a
// panic mid-stream), the status line is gone and cannot change, so we only log.
//
// It also installs the responseRecorder that inner layers read from.
func (s *Server) recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rr := &responseRecorder{ResponseWriter: w}

		defer func() {
			if v := recover(); v != nil {
				// The request ID is set by the inner requestID middleware and
				// echoed on the response header, so we can read it back here
				// even though that middleware runs "below" us.
				reqID := rr.Header().Get(headerRequestID)
				s.logger.Error("panic recovered",
					"request_id", reqID,
					"panic", v,
					"stack", string(debug.Stack()))

				if !rr.wroteHeader {
					writeError(rr, http.StatusInternalServerError, "internal_error", "internal server error")
				}
				// else: response already committed; nothing safe to do but log.
			}
		}()

		next.ServeHTTP(rr, r)
	})
}

// requestIDMiddleware assigns a correlation ID: it reuses an incoming
// X-Request-Id when present, otherwise generates one. The ID is echoed on the
// response header and attached to the context along with a request-scoped
// logger, so every downstream log line carries it.
func (s *Server) requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimSpace(r.Header.Get(headerRequestID))
		if id == "" {
			id = newRequestID()
		}
		if len(id) > 200 { // guard against absurd client-supplied values
			id = id[:200]
		}

		w.Header().Set(headerRequestID, id)

		reqLog := s.logger.With("request_id", id)
		ctx := context.WithValue(r.Context(), ctxKeyRequestID, id)
		ctx = context.WithValue(ctx, ctxKeyLogger, reqLog)

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// accessLogMiddleware logs one structured line per request after it completes:
// method, path, status, duration, and an outcome class. The request ID rides
// along automatically because we log through the request-scoped logger.
func (s *Server) accessLogMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)

		status := 0
		if rr, ok := w.(*responseRecorder); ok {
			status = rr.status
		}
		s.reqLogger(r).Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", status,
			"duration_ms", time.Since(start).Milliseconds(),
			"outcome", outcomeForStatus(status),
		)
	})
}

// authMiddleware enforces static bearer-token auth. It wraps ONLY the /v1
// routes (see Handler), so /healthz stays reachable without a key. With no keys
// configured it is a no-op ("open mode").
func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(s.cfg.APIKeys) == 0 {
			next.ServeHTTP(w, r)
			return
		}
		token, ok := bearerToken(r.Header.Get("Authorization"))
		if !ok || !keyAllowed(s.cfg.APIKeys, token) {
			writeError(w, http.StatusUnauthorized, "invalid_request_error", "invalid or missing API key")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// bearerToken extracts the token from an "Authorization: Bearer <token>" header.
func bearerToken(header string) (string, bool) {
	const prefix = "Bearer "
	if len(header) <= len(prefix) || !strings.EqualFold(header[:len(prefix)], prefix) {
		return "", false
	}
	token := strings.TrimSpace(header[len(prefix):])
	if token == "" {
		return "", false
	}
	return token, true
}

// keyAllowed reports whether presented matches any configured key. It uses a
// constant-time comparison so an attacker cannot learn a key byte-by-byte from
// response timing, and it checks every key (no early return) to keep timing
// independent of which key matched.
func keyAllowed(keys []string, presented string) bool {
	match := false
	for _, k := range keys {
		if subtle.ConstantTimeCompare([]byte(k), []byte(presented)) == 1 {
			match = true
		}
	}
	return match
}

// newRequestID returns a random 128-bit hex ID. crypto/rand.Read effectively
// never fails on a healthy system; if it somehow does, we degrade to a marker
// rather than crash a request.
func newRequestID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "req-unknown"
	}
	return hex.EncodeToString(b[:])
}

func outcomeForStatus(status int) string {
	switch {
	case status >= 500:
		return "server_error"
	case status >= 400:
		return "client_error"
	default:
		return "ok"
	}
}
