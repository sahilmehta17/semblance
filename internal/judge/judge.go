// Package judge decides response equivalence — the label c in the verified
// cache: would a cached response have been "correct" for a new prompt?
//
// Labeling runs ASYNCHRONOUSLY, off the request's critical path, through a
// bounded queue (see Labeler): a slow or unavailable judge must never slow down
// or block a user request. If the queue is full we drop the job — we simply
// learn from fewer explores, which is safe (fewer observations, never a wrong
// one).
package judge

import (
	"context"
	"strings"
)

// Judge reports whether a candidate response is equivalent to a reference one.
type Judge interface {
	Equivalent(ctx context.Context, reference, candidate string) (bool, error)
}

// ExactJudge treats responses as equivalent iff they match exactly after
// trimming surrounding whitespace. Right for short, deterministic answers.
type ExactJudge struct{}

func (ExactJudge) Equivalent(_ context.Context, reference, candidate string) (bool, error) {
	return strings.TrimSpace(reference) == strings.TrimSpace(candidate), nil
}

// LLMJudge is the pluggable semantic-equivalence path for long responses, where
// exact match is too strict. It needs a model backend, so it is left as an
// interface and stubbed for now (see wiring): the DefaultJudge falls back to
// exact match when no LLMJudge is configured.
type LLMJudge interface {
	Equivalent(ctx context.Context, reference, candidate string) (bool, error)
}

// DefaultJudge uses exact match for short responses and delegates to an
// LLMJudge for longer ones. When no LLMJudge is set it falls back to exact
// match — a conservative stub: it may label a true paraphrase as "not
// equivalent" (c=0), which only causes an extra cache entry, never a wrong hit.
type DefaultJudge struct {
	// ShortLen is the response length (in bytes) at or below which we use exact
	// match. Above it, we consult LLM (if set).
	ShortLen int
	// LLM is the optional semantic judge for long responses.
	LLM LLMJudge
}

// NewDefaultJudge returns a DefaultJudge with a sensible short-response cutoff.
func NewDefaultJudge(llm LLMJudge) *DefaultJudge {
	return &DefaultJudge{ShortLen: 256, LLM: llm}
}

func (d *DefaultJudge) Equivalent(ctx context.Context, reference, candidate string) (bool, error) {
	longer := len(reference)
	if len(candidate) > longer {
		longer = len(candidate)
	}
	if longer <= d.ShortLen || d.LLM == nil {
		return ExactJudge{}.Equivalent(ctx, reference, candidate)
	}
	return d.LLM.Equivalent(ctx, reference, candidate)
}
