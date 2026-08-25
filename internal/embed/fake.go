package embed

import (
	"context"
	"hash/fnv"
	"strings"
)

// Fake is a deterministic, offline Embedder for tests. It builds a hashed
// bag-of-words vector: each whitespace token is hashed to a dimension and adds
// +1 or -1 there (the sign comes from a second hash bit, so unrelated tokens
// tend to cancel rather than all pointing the same way).
//
// This makes similarity RELATIONSHIPS controllable in tests without any network
// or model:
//
//   - identical or reordered token sets → cosine 1.0 (a clean "paraphrase",
//     since bag-of-words ignores word order);
//   - heavy token overlap → high cosine;
//   - disjoint tokens → cosine near 0 (an "adversarial near-miss" is built by
//     mixing shared and distinguishing tokens to land between).
//
// It is NOT a semantic model — it knows nothing about meaning — but that is
// exactly what we want for deterministic cache tests.
type Fake struct {
	dim int
}

// NewFake returns a Fake with the given dimensionality (default 256 if dim<=0).
// A larger dim reduces accidental hash collisions between distinct tokens.
func NewFake(dim int) *Fake {
	if dim <= 0 {
		dim = 256
	}
	return &Fake{dim: dim}
}

func (f *Fake) Dimensions() int { return f.dim }

// Embed returns tokens=0: the fake makes no API call, so there is no real token
// count to report, and estimating one would corrupt cost accounting.
func (f *Fake) Embed(_ context.Context, text string) ([]float32, int, error) {
	v := make([]float32, f.dim)
	for _, tok := range strings.Fields(strings.ToLower(text)) {
		h := fnv.New32a()
		_, _ = h.Write([]byte(tok))
		sum := h.Sum32()
		idx := sum % uint32(f.dim)
		if sum&0x80000000 != 0 { // top bit selects the sign
			v[idx]--
		} else {
			v[idx]++
		}
	}
	return v, 0, nil
}
