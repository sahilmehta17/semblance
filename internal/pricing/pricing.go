// Package pricing loads a committed price table and turns token counts into
// dollars. Costs are always computed from real usage counts (never estimated
// token counts), so the numbers trace to the provider's own accounting.
package pricing

import (
	"encoding/json"
	"fmt"
	"os"
)

// ModelPrice is per-million-token pricing for one model. Chat models use prompt
// and completion; embedding models use embed.
type ModelPrice struct {
	PromptPer1M     float64 `json:"prompt_per_1m"`
	CompletionPer1M float64 `json:"completion_per_1m"`
	EmbedPer1M      float64 `json:"embed_per_1m"`
}

// Table is the whole price table. RetrievedDate records when the prices were
// looked up, committed alongside them so a stale table is obvious.
type Table struct {
	RetrievedDate string                `json:"retrieved_date"`
	Currency      string                `json:"currency"`
	Models        map[string]ModelPrice `json:"models"`
}

// Load reads a price table from a JSON file.
func Load(path string) (*Table, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read price table: %w", err)
	}
	var t Table
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, fmt.Errorf("parse price table: %w", err)
	}
	if t.Models == nil {
		t.Models = map[string]ModelPrice{}
	}
	return &t, nil
}

// Empty returns a table with no models (every cost is 0). Used when no price
// table is configured, so the gateway still runs (costs report as 0 rather than
// crashing).
func Empty() *Table {
	return &Table{Models: map[string]ModelPrice{}}
}

// ChatCost returns the USD cost of a chat completion given its real token usage.
// Unknown models cost 0 (and the caller may log a warning) — we never invent a
// price. Pricing is per-million tokens.
func (t *Table) ChatCost(model string, promptTokens, completionTokens int) float64 {
	p, ok := t.Models[model]
	if !ok {
		return 0
	}
	return float64(promptTokens)/1e6*p.PromptPer1M + float64(completionTokens)/1e6*p.CompletionPer1M
}

// EmbedCost returns the USD cost of an embedding call for the given real token
// usage. Unknown models cost 0.
func (t *Table) EmbedCost(model string, tokens int) float64 {
	p, ok := t.Models[model]
	if !ok {
		return 0
	}
	return float64(tokens) / 1e6 * p.EmbedPer1M
}

// Known reports whether the table has a price for the model.
func (t *Table) Known(model string) bool {
	_, ok := t.Models[model]
	return ok
}
