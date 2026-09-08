#!/usr/bin/env bash
# semblance live demo against Ollama: cold miss -> learning -> cache HIT -> metrics.
# Assumes the gateway is running on :8088 (see demo/README.md) backed by Ollama.
set -euo pipefail
GW=127.0.0.1:8088
REQ='{"model":"llama3.2:1b","temperature":0,"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}]}'
hdr() { curl -s -i "$GW/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ"; }
line() { printf '%s\n' "------------------------------------------------------------------"; }
field() { echo "$1" | grep -i "^X-Semblance-$2" | tr -d '\r' | awk '{print $2}'; }

echo "semblance — verified semantic cache (vCache, arXiv:2502.03771)"
echo "backend: Ollama llama3.2:1b   embeddings: Ollama nomic-embed-text"
printf 'health: '; curl -fs "$GW/healthz"; echo
line

echo "1) COLD MISS — empty cache, prompt seen for the first time:"
out=$(hdr "$REQ")
echo "$out" | grep -iE '^(HTTP/|X-Semblance)'
line

echo "2) OTHER TRAFFIC — different questions land near this entry as negatives,"
echo "   giving the correctness curve some variation to learn from:"
for c in Germany Spain Italy Japan; do
  q='{"model":"llama3.2:1b","temperature":0,"messages":[{"role":"user","content":"What is the capital of '"$c"'? Answer in one word."}]}'
  o=$(curl -s -i "$GW/v1/chat/completions" -H 'Content-Type: application/json' -d "$q")
  printf '   %-8s cache=%-5s sim=%s\n' "$c" "$(field "$o" Cache)" "$(field "$o" Similarity)"
  sleep 0.2
done
line

echo "3) WARM-UP — repeat the France query. The policy LEARNS: tau falls from 1.0."
echo "   It keeps EXPLORING while uncertain (verified caching is cautious by design),"
echo "   then serves HITS once it is confident enough:"
hit=""
for i in $(seq 1 60); do
  out=$(hdr "$REQ")
  cache=$(field "$out" Cache); tau=$(field "$out" Tau); sim=$(field "$out" Similarity)
  printf '   req %2d  cache=%-5s sim=%-7s tau=%s\n' "$i" "$cache" "${sim:-n/a}" "$tau"
  if [ "$cache" = "hit" ]; then hit="$out"; break; fi
  sleep 0.2
done
line

echo "4) CACHE HIT — served from the cache, backend NOT called. Full response:"
echo "$hit" | grep -iE '^(HTTP/|X-Semblance)'
printf '   body: '; echo "$hit" | tail -1
line

echo "5) METRICS MOVING (/metrics):"
curl -s "$GW/metrics" \
  | grep -E '^semblance_(requests_total|cost_usd_total|cache_entries|backend_tokens_total|embed_tokens_total|tau_count|similarity_count)' \
  | grep -v _bucket
line
echo "done. verified caching bounds an error you cannot observe live — see README."
