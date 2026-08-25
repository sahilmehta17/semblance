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
	// Embed returns the embedding of text and the number of tokens the provider
	// billed for it (0 if the provider reports none). The returned slice length
	// equals Dimensions(). Token counts come from the provider's usage field —
	// they are never estimated — so cost accounting stays honest.
	Embed(ctx context.Context, text string) (vec []float32, tokens int, err error)
	// Dimensions is the vector length this embedder produces.
	Dimensions() int
}
