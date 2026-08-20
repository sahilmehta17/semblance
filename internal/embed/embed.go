// Package embed turns prompt text into vectors for semantic matching.
//
// The gateway depends only on the Embedder interface, so the whole cache is
// testable offline with the deterministic Fake in this package; the real
// OpenAI implementation is exercised only in the live demo.
package embed

import "context"

// Embedder maps text to a dense vector. Implementations return the raw vector;
// normalization for cosine similarity is the cache's responsibility, so callers
// never have to remember to normalize.
type Embedder interface {
	// Embed returns the embedding of text. The returned slice length equals
	// Dimensions().
	Embed(ctx context.Context, text string) ([]float32, error)
	// Dimensions is the vector length this embedder produces.
	Dimensions() int
}
