package cache

// GuardedInsert implements the insertion rule from vCache Algorithm 1 (line 11):
// a new entry is added ONLY when the nearest neighbor would have been wrong for
// this prompt (neighborCorrect == false). When the neighbor was correct it
// already covers the new prompt, so inserting a near-duplicate would just waste
// space. It returns the new entry ID and true when it inserted.
//
// This is the caller-side decision, kept here so the Algorithm-1 guard lives
// (and is tested) in one place; the gateway calls this rather than Insert
// directly. See DECISIONS.md for why we follow Algorithm 1 over the paper's
// unconditional Eq. 5.
func GuardedInsert(s Store, bucket string, vec []float32, resp StoredResponse, neighborCorrect bool) (string, bool) {
	if neighborCorrect {
		return "", false
	}
	return s.Insert(bucket, vec, resp), true
}
