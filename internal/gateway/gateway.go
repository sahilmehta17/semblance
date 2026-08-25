// Package gateway wires the HTTP surface of semblance: routing plus the
// request handlers. Keeping this out of main() lets tests exercise the routes
// with httptest and no real network or process.
package gateway

import (
	"log/slog"
	"math/rand"
	"net/http"
	"sync/atomic"

	"github.com/sahilmehta17/semblance/internal/budget"
	"github.com/sahilmehta17/semblance/internal/cache"
	"github.com/sahilmehta17/semblance/internal/config"
	"github.com/sahilmehta17/semblance/internal/embed"
	"github.com/sahilmehta17/semblance/internal/judge"
	"github.com/sahilmehta17/semblance/internal/metrics"
	"github.com/sahilmehta17/semblance/internal/policy"
	"github.com/sahilmehta17/semblance/internal/pricing"
)

// Server holds the dependencies every handler shares. The verified-cache
// dependencies (embedder, store, policy, labeler) are injectable via options so
// tests can wire deterministic fakes and production wires real implementations.
type Server struct {
	cfg    *config.Config
	logger *slog.Logger
	client *http.Client

	embedder embed.Embedder // nil disables caching (pure passthrough)
	store    cache.Store
	policy   *policy.Policy
	labeler  *judge.Labeler
	metrics  *metrics.Metrics
	prices   *pricing.Table
	budget   *budget.Tracker

	bypassTotal atomic.Int64
}

// Option customizes a Server at construction (used mainly by tests).
type Option func(*Server)

// WithEmbedder injects the embedder (e.g. a deterministic fake in tests).
func WithEmbedder(e embed.Embedder) Option { return func(s *Server) { s.embedder = e } }

// WithStore injects the cache store.
func WithStore(st cache.Store) Option { return func(s *Server) { s.store = st } }

// WithPolicy injects a fully-built policy (e.g. with a deterministic RNG).
func WithPolicy(p *policy.Policy) Option { return func(s *Server) { s.policy = p } }

// WithLabeler injects the async labeler.
func WithLabeler(l *judge.Labeler) Option { return func(s *Server) { s.labeler = l } }

// WithPrices injects a price table (tests use a known one to check cost math).
func WithPrices(p *pricing.Table) Option { return func(s *Server) { s.prices = p } }

// stdRand adapts the standard library's concurrency-safe global RNG to
// policy.Rand for production use.
type stdRand struct{}

func (stdRand) Float64() float64 { return rand.Float64() }

// New builds a Server with production defaults, then applies options. The HTTP
// client has no global timeout; each upstream call is bounded by a per-request
// context so streaming is not cut short by a client-wide deadline.
func New(cfg *config.Config, logger *slog.Logger, opts ...Option) *Server {
	if len(cfg.APIKeys) == 0 {
		logger.Warn("no API keys configured (SEMBLANCE_API_KEYS); /v1 authentication is DISABLED (open mode)")
	}
	s := &Server{
		cfg:    cfg,
		logger: logger,
		client: &http.Client{},
	}
	// Apply options first, then fill only the dependencies still unset — so an
	// injected store/policy/labeler never triggers construction of a default
	// one (which for the labeler would leak goroutines).
	for _, opt := range opts {
		opt(s)
	}
	if s.store == nil {
		s.store = cache.NewMemoryStore(cfg.CacheCapacity, cfg.MaxObservations, cfg.CacheShards, 1)
	}
	if s.policy == nil {
		s.policy = policy.NewPolicy(cfg.Delta, stdRand{})
	}
	if s.labeler == nil {
		s.labeler = judge.NewLabeler(judge.NewDefaultJudge(nil), cfg.JudgeQueueSize, cfg.JudgeWorkers, logger)
	}
	if s.metrics == nil {
		s.metrics = metrics.New()
	}
	if s.prices == nil {
		// A missing/empty price table is a warning, not a fatal: the gateway
		// still runs and simply reports costs of 0 for unpriced models.
		if p, err := pricing.Load(cfg.PriceTablePath); err != nil {
			logger.Warn("price table not loaded; costs will report 0", "path", cfg.PriceTablePath, "err", err)
			s.prices = pricing.Empty()
		} else {
			s.prices = p
			logger.Info("price table loaded", "path", cfg.PriceTablePath, "retrieved_date", p.RetrievedDate, "models", len(p.Models))
		}
	}
	if s.budget == nil {
		s.budget = budget.New(cfg.BudgetCents)
	}

	// The configured delta is the TARGET; realized error is not observable live
	// (see metrics package doc / DECISIONS.md). Expose the target only.
	s.metrics.DeltaTarget.Set(cfg.Delta)
	// Gauges evaluated on each scrape, so they are correct even when idle.
	s.metrics.RegisterCacheEntriesFunc(func() float64 { return float64(s.store.Len()) })
	s.metrics.RegisterJudgeQueueFunc(func() float64 { return float64(s.labeler.QueueDepth()) })

	if s.embedder == nil {
		logger.Warn("no embedder configured (set OPENAI_API_KEY); semantic caching DISABLED (passthrough only)")
	}
	return s
}

// BypassTotal returns the number of requests that skipped the cache. This is a
// placeholder counter until Step 9 replaces it with a Prometheus metric.
func (s *Server) BypassTotal() int64 { return s.bypassTotal.Load() }

// Close releases background resources (the async labeler's workers). Call it
// during graceful shutdown, after the HTTP server has stopped accepting new
// requests so nothing submits to a closed queue.
func (s *Server) Close() {
	if s.labeler != nil {
		s.labeler.Close()
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
	// The /v1 subtree, guarded by auth then budget (auth sets the key on the
	// context; budget reads it). Order: auth(budget(v1)).
	v1 := http.NewServeMux()
	v1.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	authedV1 := s.authMiddleware(s.budgetMiddleware(v1))

	// Root mux: unauthenticated health and metrics, plus the guarded /v1 subtree.
	// /metrics is unauthenticated like /healthz so a Prometheus scraper needs no
	// key (a common, deliberate choice; noted in DECISIONS.md).
	root := http.NewServeMux()
	root.HandleFunc("GET /healthz", s.handleHealthz)
	root.Handle("GET /metrics", s.metrics.Handler())
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
