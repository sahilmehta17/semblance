// Command semblance is an OpenAI-compatible LLM gateway that implements the
// vCache verified-semantic-caching algorithm (Schroeder et al., ICLR 2026,
// arXiv:2502.03771).
//
// This file is only the process lifecycle: load config, build the logger,
// start the HTTP server, and shut it down cleanly on a termination signal.
// Request handling lives in the internal packages; keeping main.go thin makes
// the startup/shutdown story easy to read in one screen.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/sahilmehta17/semblance/internal/config"
	"github.com/sahilmehta17/semblance/internal/gateway"
)

func main() {
	// run() holds the real logic and returns an error instead of calling
	// os.Exit directly. This is a common Go pattern: os.Exit skips deferred
	// functions, so we confine it to exactly one place (here) and let run()
	// use normal defer/return everywhere else.
	if err := run(); err != nil {
		// The logger may not exist yet if config failed, so fall back to the
		// default slog logger for this last-resort message.
		slog.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	// Structured JSON logs to stdout. In Kubernetes this is what the log
	// collector scrapes; JSON keeps fields queryable instead of regex-parsed.
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: cfg.LogLevel,
	}))
	slog.SetDefault(logger)

	// signal.NotifyContext gives us a context that is cancelled when the
	// process receives SIGINT (Ctrl-C) or SIGTERM (what Kubernetes sends to
	// stop a pod). Everything downstream watches this one context to know when
	// to wind down. `stop` releases the signal handler; defer it so we don't
	// leak it.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           gateway.New(cfg, logger).Handler(),
		ReadHeaderTimeout: 10 * time.Second, // basic slow-client protection
	}

	// Run ListenAndServe in a goroutine so main can block on the context
	// instead. The buffered channel of size 1 lets the goroutine report a
	// startup failure (e.g. port already in use) without blocking if no one is
	// reading yet.
	serverErr := make(chan error, 1)
	go func() {
		logger.Info("server starting", "addr", cfg.ListenAddr)
		// ListenAndServe always returns a non-nil error. On a clean shutdown
		// it returns http.ErrServerClosed, which is expected and not a failure.
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
		}
	}()

	// Block until either the server crashes or we get a termination signal.
	select {
	case err := <-serverErr:
		return err
	case <-ctx.Done():
		logger.Info("shutdown signal received, draining connections")
	}

	// Graceful shutdown: stop accepting new connections and give in-flight
	// requests up to ShutdownTimeout to finish. We use context.Background()
	// (not ctx, which is already cancelled) with a fresh timeout.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		return err
	}
	logger.Info("server stopped cleanly")
	return nil
}
