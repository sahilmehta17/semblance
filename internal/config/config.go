// Package config loads and validates the gateway's runtime configuration from
// environment variables.
//
// The design rule from the build brief is "fail fast": if the environment is
// misconfigured we want the process to refuse to start with a clear error,
// rather than boot into a half-working state and fail later on a live request.
// So Load() returns an error for anything it cannot make sense of, and main()
// turns that into an immediate non-zero exit.
package config

import (
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"strings"
	"time"
)

// Config holds every setting the gateway needs. It starts small; later build
// steps (backend passthrough, auth, policy) add fields here. Keeping all
// configuration in one struct means the rest of the code receives a single
// read-only value and never reaches for os.Getenv on its own.
type Config struct {
	// ListenAddr is the host:port the HTTP server binds to, e.g. ":8080".
	ListenAddr string

	// LogLevel controls slog verbosity. Parsed from a human string like
	// "info" or "debug" into slog's numeric level.
	LogLevel slog.Level

	// ShutdownTimeout bounds how long we wait for in-flight requests to
	// finish during graceful shutdown before forcing the process down.
	ShutdownTimeout time.Duration

	// BackendBaseURL is the OpenAI-compatible base URL of the upstream model
	// server (e.g. Ollama at http://localhost:11434/v1). It includes the "/v1"
	// segment, matching the OpenAI SDK's base_url convention; handlers append
	// the specific path such as "/chat/completions".
	BackendBaseURL string

	// BackendTimeout bounds a single upstream request. Generous by default
	// because model completions can be slow.
	BackendTimeout time.Duration

	// APIKeys are the accepted static bearer tokens for the /v1 routes. If
	// empty, authentication is disabled ("open mode") — convenient for local
	// dev, and warned about at startup.
	APIKeys []string
}

// Load reads configuration from the environment and validates it.
//
// Every value has a safe default, so an empty environment still yields a
// runnable gateway. Validation only rejects values that are present but
// malformed — an unknown log level, an unparseable duration — because those
// signal operator mistakes we want surfaced at startup.
func Load() (*Config, error) {
	cfg := &Config{
		// getenv is a tiny helper (below) that returns the fallback when the
		// variable is unset or empty. This "default via helper" pattern keeps
		// the struct literal readable instead of a wall of if-statements.
		ListenAddr: getenv("SEMBLANCE_LISTEN_ADDR", ":8080"),
	}

	level, err := parseLogLevel(getenv("SEMBLANCE_LOG_LEVEL", "info"))
	if err != nil {
		return nil, err
	}
	cfg.LogLevel = level

	timeout, err := time.ParseDuration(getenv("SEMBLANCE_SHUTDOWN_TIMEOUT", "15s"))
	if err != nil {
		// %w wraps the underlying error so callers can inspect it with
		// errors.Is/As if they ever need to; it also keeps the original
		// message. This is the standard Go way to add context to an error.
		return nil, fmt.Errorf("SEMBLANCE_SHUTDOWN_TIMEOUT: %w", err)
	}
	cfg.ShutdownTimeout = timeout

	// Trim a trailing slash so handlers can append paths without doubling up.
	cfg.BackendBaseURL = strings.TrimRight(getenv("SEMBLANCE_BACKEND_URL", "http://localhost:11434/v1"), "/")
	if err := validateHTTPURL(cfg.BackendBaseURL); err != nil {
		return nil, fmt.Errorf("SEMBLANCE_BACKEND_URL: %w", err)
	}

	backendTimeout, err := time.ParseDuration(getenv("SEMBLANCE_BACKEND_TIMEOUT", "120s"))
	if err != nil {
		return nil, fmt.Errorf("SEMBLANCE_BACKEND_TIMEOUT: %w", err)
	}
	cfg.BackendTimeout = backendTimeout

	cfg.APIKeys = parseKeys(getenv("SEMBLANCE_API_KEYS", ""))

	return cfg, nil
}

// parseKeys splits a comma-separated key list, trimming whitespace and dropping
// empty entries. An unset or blank variable yields a nil slice (open mode).
func parseKeys(raw string) []string {
	var keys []string
	for _, k := range strings.Split(raw, ",") {
		if k = strings.TrimSpace(k); k != "" {
			keys = append(keys, k)
		}
	}
	return keys
}

// validateHTTPURL rejects anything that is not an absolute http(s) URL with a
// host. Catching this at startup means a typo in the backend address fails the
// process immediately instead of on the first proxied request.
func validateHTTPURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return err
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("must be http or https, got %q", u.Scheme)
	}
	if u.Host == "" {
		return fmt.Errorf("missing host in %q", raw)
	}
	return nil
}

// getenv returns the environment variable named by key, or fallback if the
// variable is unset or the empty string.
func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// parseLogLevel maps the usual level names onto slog levels. We validate here
// rather than silently defaulting so a typo like "warm" fails loudly.
func parseLogLevel(s string) (slog.Level, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "debug":
		return slog.LevelDebug, nil
	case "info":
		return slog.LevelInfo, nil
	case "warn", "warning":
		return slog.LevelWarn, nil
	case "error":
		return slog.LevelError, nil
	default:
		return 0, fmt.Errorf("SEMBLANCE_LOG_LEVEL: unknown level %q (want debug|info|warn|error)", s)
	}
}
