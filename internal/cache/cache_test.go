package cache

import (
	"context"
	"math/rand"
	"sync"
	"testing"

	"github.com/sahilmehta17/semblance/internal/embed"
)

var fake = embed.NewFake(256)

func emb(t *testing.T, text string) []float32 {
	t.Helper()
	v, err := fake.Embed(context.Background(), text)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	return v
}

func resp(body string) StoredResponse {
	return StoredResponse{Body: []byte(body), ContentType: "application/json"}
}

// TestParaphraseHitsRightEntry: a reordered paraphrase finds the correct entry
// with near-perfect similarity, and a decoy entry is not chosen.
func TestParaphraseHitsRightEntry(t *testing.T) {
	s := NewMemoryStore(100, 50, 4, 1)
	bucket := BucketKey("model", "sys", "0", "1")

	idFrance := s.Insert(bucket, emb(t, "please tell me the capital of france"), resp("paris"))
	_ = s.Insert(bucket, emb(t, "what is the tallest mountain in nepal"), resp("everest"))

	// Same tokens, reordered → cosine 1.0 with the France entry.
	m, ok := s.Nearest(bucket, emb(t, "the capital of france please tell me"))
	if !ok {
		t.Fatal("expected a match")
	}
	if m.EntryID != idFrance {
		t.Errorf("matched %s, want France entry %s", m.EntryID, idFrance)
	}
	if m.Similarity < 0.99 {
		t.Errorf("paraphrase similarity = %.4f, want ~1.0", m.Similarity)
	}
}

// TestDifferentSystemPromptNeverMatches: a request in a different bucket (only
// the system prompt differs) must not see the other bucket's entries, no matter
// how identical the user turn is.
func TestDifferentSystemPromptNeverMatches(t *testing.T) {
	s := NewMemoryStore(100, 50, 4, 1)
	text := "please tell me the capital of france"

	helpful := BucketKey("model", "You are a helpful assistant", "0", "1")
	evil := BucketKey("model", "You are a sarcastic assistant", "0", "1")

	s.Insert(helpful, emb(t, text), resp("paris"))

	// Identical user turn, different system prompt → different bucket → no match.
	if _, ok := s.Nearest(evil, emb(t, text)); ok {
		t.Error("a different system prompt must never hit another bucket's entry")
	}
}

// TestAdversarialNearMissDoesNotWronglyMatch: a query that shares some tokens
// but is genuinely different lands on the closest entry but at a materially
// lower similarity than a true paraphrase — so the policy can tell them apart.
func TestAdversarialNearMissDoesNotWronglyMatch(t *testing.T) {
	s := NewMemoryStore(100, 50, 4, 1)
	bucket := BucketKey("model", "sys", "0", "1")

	idFrance := s.Insert(bucket, emb(t, "please tell me the capital of france"), resp("paris"))
	s.Insert(bucket, emb(t, "what is the tallest mountain in nepal"), resp("everest"))

	// Shares "the capital of ... please" but asks about Germany.
	m, ok := s.Nearest(bucket, emb(t, "the capital of germany please"))
	if !ok {
		t.Fatal("expected a nearest entry")
	}
	if m.EntryID != idFrance {
		t.Errorf("nearest = %s, want France entry (closest by overlap)", m.EntryID)
	}
	if m.Similarity >= 0.9 {
		t.Errorf("near-miss similarity = %.4f, want < 0.9 (must be clearly below a paraphrase)", m.Similarity)
	}
}

// TestGuardedInsert: Algorithm 1 inserts only when the neighbor was wrong.
func TestGuardedInsert(t *testing.T) {
	s := NewMemoryStore(100, 50, 1, 1)
	bucket := BucketKey("m", "s", "0", "1")
	vec := emb(t, "hello world")

	if _, inserted := GuardedInsert(s, bucket, vec, resp("x"), true); inserted {
		t.Error("must NOT insert when neighbor was correct (c=1)")
	}
	if s.Len() != 0 {
		t.Errorf("Len = %d, want 0 after guarded no-op", s.Len())
	}

	if _, inserted := GuardedInsert(s, bucket, vec, resp("x"), false); !inserted {
		t.Error("must insert when neighbor was wrong (c=0)")
	}
	if s.Len() != 1 {
		t.Errorf("Len = %d, want 1 after guarded insert", s.Len())
	}
}

// TestLRUEviction: with capacity 2 in a single shard, inserting a third entry
// evicts the least-recently-used one, and a refreshed entry survives.
func TestLRUEviction(t *testing.T) {
	s := NewMemoryStore(2, 50, 1, 1) // capacity 2, single shard for determinism
	bucket := BucketKey("m", "s", "0", "1")

	// Disjoint-token texts → near-orthogonal vectors, each identifiable by
	// querying its own vector (self-cosine 1.0).
	va, vb, vc := emb(t, "alpha alpha"), emb(t, "bravo bravo"), emb(t, "charlie charlie")
	idA := s.Insert(bucket, va, resp("A"))
	idB := s.Insert(bucket, vb, resp("B"))

	// Touch A so B becomes the least-recently-used.
	if m, _ := s.Nearest(bucket, va); m.EntryID != idA {
		t.Fatalf("sanity: nearest to A = %s, want %s", m.EntryID, idA)
	}

	idC := s.Insert(bucket, vc, resp("C")) // should evict B
	if s.Len() != 2 {
		t.Fatalf("Len = %d, want 2 after eviction", s.Len())
	}

	// B is gone: its own vector no longer resolves to idB.
	if m, ok := s.Nearest(bucket, vb); ok && m.EntryID == idB {
		t.Error("B should have been evicted as least-recently-used")
	}
	// A and C survive.
	if m, _ := s.Nearest(bucket, va); m.EntryID != idA {
		t.Error("A should have survived (was refreshed)")
	}
	if m, _ := s.Nearest(bucket, vc); m.EntryID != idC {
		t.Error("C should be present (just inserted)")
	}
}

// TestObservationCapReservoir: after far more observations than the cap, the
// entry holds exactly maxObs of them.
func TestObservationCapReservoir(t *testing.T) {
	const maxObs = 50
	s := NewMemoryStore(10, maxObs, 1, 1)
	bucket := BucketKey("m", "s", "0", "1")
	id := s.Insert(bucket, emb(t, "hello world"), resp("x"))

	for i := 0; i < 1000; i++ {
		s.Observe(bucket, id, 0.8, i%2 == 0)
	}

	m, ok := s.Nearest(bucket, emb(t, "hello world"))
	if !ok {
		t.Fatal("entry missing")
	}
	if len(m.Observations) != maxObs {
		t.Errorf("observation count = %d, want capped at %d", len(m.Observations), maxObs)
	}
}

// TestConcurrentAccessNoRace hammers the store from many goroutines across
// several buckets. Its value is under `go test -race`: any data race fails.
func TestConcurrentAccessNoRace(t *testing.T) {
	s := NewMemoryStore(500, 32, 16, 7)
	buckets := make([]string, 8)
	for i := range buckets {
		buckets[i] = BucketKey("m", "s", "0", string(rune('a'+i)))
	}

	var wg sync.WaitGroup
	for g := 0; g < 32; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(g))) // per-goroutine, no shared RNG
			for i := 0; i < 500; i++ {
				b := buckets[rng.Intn(len(buckets))]
				vec := emb(t, "token"+string(rune('a'+rng.Intn(20))))
				switch rng.Intn(3) {
				case 0:
					id := s.Insert(b, vec, resp("r"))
					s.Observe(b, id, rng.Float64(), rng.Intn(2) == 0)
				case 1:
					if m, ok := s.Nearest(b, vec); ok {
						s.Observe(b, m.EntryID, rng.Float64(), true)
					}
				default:
					_, _ = s.Nearest(b, vec)
				}
			}
		}(g)
	}
	wg.Wait()

	if s.Len() == 0 {
		t.Error("expected some entries to survive")
	}
}

func TestBucketKeyLengthPrefixing(t *testing.T) {
	// Without length-prefixing these would collide.
	if BucketKey("a", "bc") == BucketKey("ab", "c") {
		t.Error("BucketKey must not collide on delimiter-ambiguous parts")
	}
	// Deterministic.
	if BucketKey("x", "y") != BucketKey("x", "y") {
		t.Error("BucketKey must be deterministic")
	}
}
