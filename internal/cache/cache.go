// Package cache is the verified semantic cache: it stores past prompt/response
// pairs and, for a new prompt, finds the nearest prior entry within an
// exact-match bucket so the policy can decide whether to reuse its answer.
//
// Design points the build brief calls out (and that this package resolves as
// "systems gaps" the paper leaves open):
//
//   - Vectors are normalized at insert, so cosine similarity is a plain dot
//     product at query time.
//   - Semantic matching is scoped to an exact-match BUCKET keyed on model,
//     system prompt, sampling params, and all prior turns; only the final user
//     turn is matched semantically. Two requests that differ in any bucket
//     input can never match each other.
//   - Guarded insert (paper Algorithm 1): a new entry is added only when the
//     nearest neighbor would have been WRONG (c=0); a correct neighbor already
//     covers the new prompt.
//   - Bounded memory via per-shard LRU eviction, and a per-entry observation
//     cap via reservoir sampling.
//   - Thread-safe under concurrent load via a sharded lock (concurrency is one
//     of the gaps we claim to fix; see the -race test).
package cache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"

	"github.com/sahilmehta17/semblance/internal/policy"
)

// StoredResponse is what we serve on a cache hit: the response body bytes and
// its content type. Status is assumed 200 (only successful completions are
// cached).
type StoredResponse struct {
	Body        []byte
	ContentType string
}

// Match is a race-safe SNAPSHOT of the nearest entry. It intentionally returns
// copies (observations, response bytes) rather than a live pointer into the
// store, so the caller can read them without holding a lock and without risking
// a data race against concurrent writers.
type Match struct {
	EntryID      string
	Similarity   float64
	Observations []policy.Observation
	Response     StoredResponse
}

// Store is the cache interface the gateway depends on. All methods are safe for
// concurrent use.
type Store interface {
	// Nearest returns a snapshot of the highest-cosine entry in the bucket, or
	// ok=false if the bucket is empty. It normalizes vec internally.
	Nearest(bucket string, vec []float32) (Match, bool)

	// Observe appends an observation (s, c) to an existing entry, applying the
	// reservoir cap, and marks the entry recently used. A no-op if the entry is
	// gone (e.g. evicted).
	Observe(bucket, entryID string, s float64, c bool) bool

	// Insert adds a new entry (vec is normalized on the way in) and returns its
	// ID. Guarded-insert is the CALLER's decision (Algorithm 1); Insert itself
	// just stores. May trigger LRU eviction.
	Insert(bucket string, vec []float32, resp StoredResponse) string

	// Len returns the total number of entries across all shards.
	Len() int
}

// BucketKey hashes an ordered list of parts into a bucket identifier. It
// length-prefixes each part so that concatenation is unambiguous — ["a","bc"]
// and ["ab","c"] must produce different keys. The gateway assembles the parts
// (model, system prompt, temperature, top_p, prior messages).
func BucketKey(parts ...string) string {
	h := sha256.New()
	var lenbuf [8]byte
	for _, p := range parts {
		binary.BigEndian.PutUint64(lenbuf[:], uint64(len(p)))
		_, _ = h.Write(lenbuf[:])
		_, _ = h.Write([]byte(p))
	}
	return hex.EncodeToString(h.Sum(nil))
}
