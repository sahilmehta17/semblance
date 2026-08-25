package gateway

import (
	"encoding/json"

	"github.com/sahilmehta17/semblance/internal/openai"
)

// centsPerUSD converts the pricing package's dollar costs into the cents the
// budget tracker uses.
const centsPerUSD = 100.0

// parseUsage pulls the token usage out of a completion response body (zero-value
// on any parse failure — we never invent counts).
func parseUsage(body []byte) openai.Usage {
	var r openai.ChatCompletionResponse
	_ = json.Unmarshal(body, &r)
	return r.Usage
}

// recordEmbedUsage records embedding tokens + cost and bills the key's budget.
// Charged on every cacheable request (an embedding happens on hits and misses
// alike, to find the neighbor).
func (s *Server) recordEmbedUsage(model, key string, tokens int) {
	if tokens > 0 {
		s.metrics.EmbedTokens.Add(float64(tokens))
	}
	if cost := s.prices.EmbedCost(model, tokens); cost > 0 {
		s.metrics.CostUSD.WithLabelValues("embedding").Add(cost)
		s.budget.Add(key, cost*centsPerUSD)
	}
	s.updateBudgetGauge(key)
}

// recordBackendUsage records backend tokens + spent cost and bills the budget.
// Charged on an EXPLORE (a real backend call happened).
func (s *Server) recordBackendUsage(model, key string, body []byte) {
	u := parseUsage(body)
	if u.PromptTokens > 0 {
		s.metrics.BackendTokens.WithLabelValues("prompt").Add(float64(u.PromptTokens))
	}
	if u.CompletionTokens > 0 {
		s.metrics.BackendTokens.WithLabelValues("completion").Add(float64(u.CompletionTokens))
	}
	if cost := s.prices.ChatCost(model, u.PromptTokens, u.CompletionTokens); cost > 0 {
		s.metrics.CostUSD.WithLabelValues("spent").Add(cost)
		s.budget.Add(key, cost*centsPerUSD)
	}
	s.updateBudgetGauge(key)
}

// recordSavedCost adds the MODELED cost of the backend call that a hit avoided:
// the served cached answer's own token usage, priced at the request's model.
// This is the honest model for "the call that didn't happen" — a paraphrase of a
// cached prompt would cost about what the cached generation did. It is NOT
// billed to the budget (no real spend occurred).
func (s *Server) recordSavedCost(model string, body []byte) {
	u := parseUsage(body)
	if cost := s.prices.ChatCost(model, u.PromptTokens, u.CompletionTokens); cost > 0 {
		s.metrics.CostUSD.WithLabelValues("saved").Add(cost)
	}
}

// updateBudgetGauge refreshes the per-key remaining-budget gauge.
func (s *Server) updateBudgetGauge(key string) {
	if s.budget.Enabled() {
		s.metrics.BudgetCents.WithLabelValues(key).Set(s.budget.RemainingCents(key))
	}
}
