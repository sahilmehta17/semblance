package config

import (
	"log/slog"
	"testing"
	"time"
)

// TestLoadDefaults verifies that an empty environment yields the documented
// defaults. t.Setenv automatically restores the previous value when the test
// ends, so tests stay isolated from each other and from the real environment.
func TestLoadDefaults(t *testing.T) {
	// Explicitly clear the vars this test cares about so a value in the
	// developer's shell can't leak in and change the result.
	t.Setenv("SEMBLANCE_LISTEN_ADDR", "")
	t.Setenv("SEMBLANCE_LOG_LEVEL", "")
	t.Setenv("SEMBLANCE_SHUTDOWN_TIMEOUT", "")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned error on defaults: %v", err)
	}
	if cfg.ListenAddr != ":8080" {
		t.Errorf("ListenAddr = %q, want %q", cfg.ListenAddr, ":8080")
	}
	if cfg.LogLevel != slog.LevelInfo {
		t.Errorf("LogLevel = %v, want %v", cfg.LogLevel, slog.LevelInfo)
	}
	if cfg.ShutdownTimeout != 15*time.Second {
		t.Errorf("ShutdownTimeout = %v, want %v", cfg.ShutdownTimeout, 15*time.Second)
	}
	if cfg.BackendBaseURL != "http://localhost:11434/v1" {
		t.Errorf("BackendBaseURL = %q, want default Ollama URL", cfg.BackendBaseURL)
	}
	if cfg.BackendTimeout != 120*time.Second {
		t.Errorf("BackendTimeout = %v, want %v", cfg.BackendTimeout, 120*time.Second)
	}
}

func TestLoadTrimsBackendTrailingSlash(t *testing.T) {
	t.Setenv("SEMBLANCE_BACKEND_URL", "http://example.com:1234/v1/")
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if cfg.BackendBaseURL != "http://example.com:1234/v1" {
		t.Errorf("BackendBaseURL = %q, want trailing slash trimmed", cfg.BackendBaseURL)
	}
}

func TestLoadOverrides(t *testing.T) {
	t.Setenv("SEMBLANCE_LISTEN_ADDR", "127.0.0.1:9999")
	t.Setenv("SEMBLANCE_LOG_LEVEL", "DEBUG") // case-insensitive
	t.Setenv("SEMBLANCE_SHUTDOWN_TIMEOUT", "30s")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned error: %v", err)
	}
	if cfg.ListenAddr != "127.0.0.1:9999" {
		t.Errorf("ListenAddr = %q, want %q", cfg.ListenAddr, "127.0.0.1:9999")
	}
	if cfg.LogLevel != slog.LevelDebug {
		t.Errorf("LogLevel = %v, want %v", cfg.LogLevel, slog.LevelDebug)
	}
	if cfg.ShutdownTimeout != 30*time.Second {
		t.Errorf("ShutdownTimeout = %v, want %v", cfg.ShutdownTimeout, 30*time.Second)
	}
}

// TestLoadRejectsBadValues confirms the fail-fast contract: malformed values
// must produce an error, not a silent fallback.
func TestLoadRejectsBadValues(t *testing.T) {
	tests := []struct {
		name string
		env  map[string]string
	}{
		{"bad log level", map[string]string{"SEMBLANCE_LOG_LEVEL": "warm"}},
		{"bad duration", map[string]string{"SEMBLANCE_SHUTDOWN_TIMEOUT": "soon"}},
		{"backend not a url", map[string]string{"SEMBLANCE_BACKEND_URL": "://nope"}},
		{"backend wrong scheme", map[string]string{"SEMBLANCE_BACKEND_URL": "ftp://host/v1"}},
		{"backend no host", map[string]string{"SEMBLANCE_BACKEND_URL": "http:///v1"}},
		{"bad backend timeout", map[string]string{"SEMBLANCE_BACKEND_TIMEOUT": "later"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for k, v := range tt.env {
				t.Setenv(k, v)
			}
			if _, err := Load(); err == nil {
				t.Errorf("Load() = nil error, want error for %s", tt.name)
			}
		})
	}
}
