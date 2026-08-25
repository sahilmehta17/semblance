package pricing

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func approx(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("got %.12f, want %.12f", got, want)
	}
}

func TestChatCostHandComputed(t *testing.T) {
	tbl := &Table{Models: map[string]ModelPrice{
		"m": {PromptPer1M: 0.59, CompletionPer1M: 0.79},
	}}
	// 1000 prompt @ 0.59/1M = 0.00059 ; 500 completion @ 0.79/1M = 0.000395
	approx(t, tbl.ChatCost("m", 1000, 500), 0.00059+0.000395)
	// unknown model -> 0 (never invented)
	approx(t, tbl.ChatCost("unknown", 1000, 500), 0)
	// zero tokens -> 0
	approx(t, tbl.ChatCost("m", 0, 0), 0)
}

func TestEmbedCostHandComputed(t *testing.T) {
	tbl := &Table{Models: map[string]ModelPrice{
		"e": {EmbedPer1M: 0.02},
	}}
	// 1_000_000 tokens @ 0.02/1M = exactly 0.02
	approx(t, tbl.EmbedCost("e", 1_000_000), 0.02)
	approx(t, tbl.EmbedCost("e", 1000), 0.00002)
	approx(t, tbl.EmbedCost("unknown", 1000), 0)
}

func TestLoadFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prices.json")
	os.WriteFile(path, []byte(`{"retrieved_date":"2026-08-25","currency":"USD",
		"models":{"m":{"prompt_per_1m":1.0,"completion_per_1m":2.0}}}`), 0o644)

	tbl, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if tbl.RetrievedDate != "2026-08-25" {
		t.Errorf("retrieved_date = %q", tbl.RetrievedDate)
	}
	if !tbl.Known("m") {
		t.Error("model m should be known")
	}
	approx(t, tbl.ChatCost("m", 1_000_000, 1_000_000), 3.0) // 1.0 + 2.0
}

func TestLoadMissingFile(t *testing.T) {
	if _, err := Load("/no/such/prices.json"); err == nil {
		t.Error("expected error for missing file")
	}
}
