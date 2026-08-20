package cache

import (
	"container/list"
	"hash/fnv"
	"math"
	"math/rand"
	"strconv"
	"sync"

	"github.com/sahilmehta17/semblance/internal/policy"
)

// entry is one cached prompt/response with its learned observations. Fields are
// only touched while holding the owning shard's lock.
type entry struct {
	id     string
	bucket string
	vec    []float32            // normalized at insert
	resp   StoredResponse       // immutable after insert (safe to share on read)
	obs    []policy.Observation // mutated under lock (reservoir); copied on read
	seen   int64                // total observations ever offered (for reservoir)
	elem   *list.Element        // this entry's node in the shard LRU list
}

// shard owns a disjoint slice of the keyspace behind its own mutex, so requests
// to different buckets can proceed in parallel. Each shard keeps its own LRU
// list and capacity — approximating a global LRU, which is the standard trade
// for lock-striped caches.
type shard struct {
	mu       sync.Mutex
	buckets  map[string]map[string]*entry // bucket -> entryID -> entry
	lru      *list.List                   // *entry, front = most recently used
	capacity int                          // max entries in this shard
	maxObs   int                          // per-entry observation cap
	rng      *rand.Rand                   // reservoir randomness (guarded by mu)
	counter  uint64                       // entry-ID sequence
	index    int
}

// MemoryStore is the in-memory Store implementation.
type MemoryStore struct {
	shards []*shard
}

// NewMemoryStore builds a store with the given TOTAL entry capacity, per-entry
// observation cap, shard count, and RNG seed. Total capacity is split evenly
// across shards. Use shardCount=1 for deterministic LRU tests.
func NewMemoryStore(totalCapacity, maxObs, shardCount int, seed int64) *MemoryStore {
	if shardCount < 1 {
		shardCount = 1
	}
	if maxObs < 1 {
		maxObs = 1
	}
	perShard := totalCapacity / shardCount
	if perShard < 1 {
		perShard = 1
	}
	s := &MemoryStore{shards: make([]*shard, shardCount)}
	for i := range s.shards {
		s.shards[i] = &shard{
			buckets:  make(map[string]map[string]*entry),
			lru:      list.New(),
			capacity: perShard,
			maxObs:   maxObs,
			// Per-shard seed keeps runs reproducible while avoiding all shards
			// sharing an identical stream.
			rng:   rand.New(rand.NewSource(seed + int64(i))),
			index: i,
		}
	}
	return s
}

// shardFor selects the shard owning a bucket. All entries in one bucket live in
// one shard, so a bucket's nearest-neighbor scan needs only that shard's lock.
func (m *MemoryStore) shardFor(bucket string) *shard {
	h := fnv.New32a()
	_, _ = h.Write([]byte(bucket))
	return m.shards[h.Sum32()%uint32(len(m.shards))]
}

func (m *MemoryStore) Nearest(bucket string, vec []float32) (Match, bool) {
	q := normalize(vec)
	sh := m.shardFor(bucket)

	sh.mu.Lock()
	defer sh.mu.Unlock()

	entries := sh.buckets[bucket]
	if len(entries) == 0 {
		return Match{}, false
	}

	var best *entry
	bestSim := math.Inf(-1)
	for _, e := range entries {
		if len(e.vec) != len(q) {
			continue // different embedder dims; not comparable
		}
		sim := dot(q, e.vec)
		if sim > bestSim {
			bestSim, best = sim, e
		}
	}
	if best == nil {
		return Match{}, false
	}

	// Matched entry counts as recently used.
	sh.lru.MoveToFront(best.elem)

	// Snapshot under lock: copy observations (they mutate under concurrent
	// Observe); the response body is immutable after insert, so sharing it is
	// race-free.
	obsCopy := make([]policy.Observation, len(best.obs))
	copy(obsCopy, best.obs)

	return Match{
		EntryID:      best.id,
		Similarity:   bestSim,
		Observations: obsCopy,
		Response:     best.resp,
	}, true
}

func (m *MemoryStore) Observe(bucket, entryID string, s float64, c bool) bool {
	sh := m.shardFor(bucket)
	sh.mu.Lock()
	defer sh.mu.Unlock()

	entries := sh.buckets[bucket]
	e := entries[entryID]
	if e == nil {
		return false // entry gone (e.g. evicted) — observation harmlessly dropped
	}

	e.seen++
	ob := policy.Observation{S: s, C: c}
	if len(e.obs) < sh.maxObs {
		e.obs = append(e.obs, ob)
	} else {
		// Reservoir sampling (Algorithm R): keep a uniform random sample of the
		// stream in fixed space. The j-th arrival (1-indexed = seen) replaces a
		// random existing slot with probability maxObs/seen.
		j := sh.rng.Int63n(e.seen)
		if j < int64(sh.maxObs) {
			e.obs[j] = ob
		}
	}

	sh.lru.MoveToFront(e.elem)
	return true
}

func (m *MemoryStore) Insert(bucket string, vec []float32, resp StoredResponse) string {
	sh := m.shardFor(bucket)
	sh.mu.Lock()
	defer sh.mu.Unlock()

	sh.counter++
	id := strconv.Itoa(sh.index) + "-" + strconv.FormatUint(sh.counter, 10)
	e := &entry{
		id:     id,
		bucket: bucket,
		vec:    normalize(vec),
		resp:   resp,
	}
	e.elem = sh.lru.PushFront(e)

	if sh.buckets[bucket] == nil {
		sh.buckets[bucket] = make(map[string]*entry)
	}
	sh.buckets[bucket][id] = e

	// Evict least-recently-used entries until within capacity.
	for sh.lru.Len() > sh.capacity {
		sh.evictLRU()
	}
	return id
}

// evictLRU removes the least-recently-used entry. Caller holds sh.mu.
func (sh *shard) evictLRU() {
	back := sh.lru.Back()
	if back == nil {
		return
	}
	victim := back.Value.(*entry)
	sh.lru.Remove(back)
	if b := sh.buckets[victim.bucket]; b != nil {
		delete(b, victim.id)
		if len(b) == 0 {
			delete(sh.buckets, victim.bucket)
		}
	}
}

func (m *MemoryStore) Len() int {
	n := 0
	for _, sh := range m.shards {
		sh.mu.Lock()
		n += sh.lru.Len()
		sh.mu.Unlock()
	}
	return n
}

// normalize returns a unit-length copy of vec (or a copy of a zero vector if vec
// has zero norm). Normalizing here means cosine similarity is a dot product.
func normalize(vec []float32) []float32 {
	out := make([]float32, len(vec))
	var norm float64
	for _, x := range vec {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		return out
	}
	inv := float32(1 / math.Sqrt(norm))
	for i, x := range vec {
		out[i] = x * inv
	}
	return out
}

// dot is the inner product of two equal-length vectors. For normalized vectors
// this equals cosine similarity.
func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}
