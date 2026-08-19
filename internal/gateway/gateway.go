// Package gateway wires the HTTP surface of semblance: routing plus the
// request handlers. Keeping this out of main() lets tests exercise the routes
// with httptest and no real network or process.
package gateway

import (
	"log/slog"
	"net/http"

	"github.com/sahilmehta17/semblance/internal/config"
)

// Server holds the dependencies every handler shares: configuration, a logger,
// and the HTTP client used to talk to the upstream backend.
type Server struct {
	cfg    *config.Config
	logger *slog.Logger
	client *http.Client
}

// New builds a Server. The HTTP client intentionally has no global timeout; we
// bound each upstream call with a per-request context instead (context timeouts
// compose with client cancellation and, unlike Client.Timeout, will not fight
// streaming responses in a later step).
func New(cfg *config.Config, logger *slog.Logger) *Server {
	if len(cfg.APIKeys) == 0 {
		logger.Warn("no API keys configured (SEMBLANCE_API_KEYS); /v1 authentication is DISABLED (open mode)")
	}
	return &Server{
		cfg:    cfg,
		logger: logger,
		client: &http.Client{},
	}
}

// Handler returns the router wrapped in the middleware chain.
//
// Route/auth layout: the /v1 API sits behind authMiddleware, while /healthz
// (and any future /readyz added to the root mux) stays reachable without a key.
// The common middleware — recover, requestID, access logging — wraps everything,
// so health checks are still logged and panic-safe. Chain order, outermost to
// innermost: recover → requestID → accessLog → (auth on /v1 only) → handler.
//
// Routes use Go 1.22+ method-and-path patterns, so a wrong method on a known
// path yields 405 automatically.
func (s *Server) Handler() http.Handler {
	// The /v1 subtree, guarded by auth.
	v1 := http.NewServeMux()
	v1.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	authedV1 := s.authMiddleware(v1)

	// Root mux: unauthenticated health plus the guarded /v1 subtree.
	root := http.NewServeMux()
	root.HandleFunc("GET /healthz", s.handleHealthz)
	root.Handle("/v1/", authedV1)

	// Common middleware applied to all routes.
	return s.recoverMiddleware(
		s.requestIDMiddleware(
			s.accessLogMiddleware(root)))
}

// handleHealthz is a liveness probe: if the process can answer, it is up. It
// deliberately does not check the backend (that is readiness, a later concern).
func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}
