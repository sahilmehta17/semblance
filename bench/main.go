// Command bench is the offline replay harness for the verified semantic cache.
//
// It feeds the extracted SearchQueries JSONL (prompt/id_set/embedding) through
// the cache decision logic in order and scores the error-vs-hit-rate tradeoff
// for two arms:
//
//   - static:   the GPTCache/LiteLLM baseline — exploit iff similarity >= T.
//   - verified: our policy — exploit iff policy.Decide says so, for each delta.
//
// It uses the dataset's precomputed emb_gte vectors directly (no live embedder)
// and labels correctness from ground-truth id_set equality (no LLM judge). The
// benchmark logic is all Go; plotting is a separate presentation-only script.
//
// The go/no-go: every verified run's realized FP/n must be <= its delta. If any
// exceeds it, that is an implementation bug — the harness exits non-zero.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"time"
)

func main() {
	in := flag.String("input", "tools/prep/out/searchqueries.jsonl", "path to extracted JSONL")
	limit := flag.Int("limit", 2000, "max queries to replay (0 = all)")
	capacity := flag.Int("capacity", 0, "max cache entries (0 = unbounded, no eviction)")
	seed := flag.Int64("seed", 42, "RNG seed for the randomized policy")
	workers := flag.Int("workers", runtime.NumCPU(), "parallel scan workers")
	nmin := flag.Int("nmin", 5, "verified-policy cold-start floor (force-explore below this many observations)")
	arm := flag.String("arm", "both", "which arms to run: static | verified | both")
	outJSON := flag.String("out", "bench/results.json", "results.json output path")
	outCSV := flag.String("csv", "bench/results.csv", "flat CSV output path")
	flag.Parse()

	runStatic := *arm == "both" || *arm == "static"
	runVerified := *arm == "both" || *arm == "verified"

	records, err := loadJSONL(*in, *limit)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load error:", err)
		os.Exit(1)
	}
	if len(records) == 0 {
		fmt.Fprintln(os.Stderr, "no records loaded")
		os.Exit(1)
	}

	thresholds := []float64{0.80, 0.83, 0.86, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99}
	deltas := []float64{0.01, 0.02, 0.03, 0.05}

	fmt.Printf("bench: n=%d capacity=%s seed=%d workers=%d nmin=%d arm=%s\n",
		len(records), capacityLabel(*capacity), *seed, *workers, *nmin, *arm)

	var results []runResult
	if runStatic {
		for _, t := range thresholds {
			start := time.Now()
			r := runArm(records, "static", t, *seed, *capacity, *workers, *nmin)
			results = append(results, r)
			fmt.Printf("  static  T=%.2f  hit=%.4f  err=%.5f  entries=%d  (%s)\n",
				t, r.HitRate, r.ErrorRate, r.EntriesFinal, time.Since(start).Round(time.Millisecond))
		}
	}
	if runVerified {
		for _, d := range deltas {
			start := time.Now()
			r := runArm(records, "verified", d, *seed, *capacity, *workers, *nmin)
			results = append(results, r)
			mark := "OK"
			if !r.WithinDelta {
				mark = "*** EXCEEDS DELTA ***"
			}
			fmt.Printf("  verified delta=%.2f nmin=%d  hit=%.4f  err=%.5f (<= %.2f? %s)  entries=%d  (%s)\n",
				d, *nmin, r.HitRate, r.ErrorRate, d, mark, r.EntriesFinal, time.Since(start).Round(time.Millisecond))
		}
	}

	summary := resultsFile{
		Dataset:  *in,
		N:        len(records),
		Capacity: *capacity,
		Seed:     *seed,
		Workers:  *workers,
		NMin:     *nmin,
		Runs:     results,
	}

	if err := writeJSON(*outJSON, summary); err != nil {
		fmt.Fprintln(os.Stderr, "write json:", err)
		os.Exit(1)
	}
	if err := writeCSV(*outCSV, results); err != nil {
		fmt.Fprintln(os.Stderr, "write csv:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s and %s\n", *outJSON, *outCSV)

	// Go/no-go gate: fail loudly if any verified run exceeded its delta.
	var breached []runResult
	for _, r := range results {
		if r.Arm == "verified" && !r.WithinDelta {
			breached = append(breached, r)
		}
	}
	if len(breached) > 0 {
		fmt.Fprintln(os.Stderr, "\nGO/NO-GO FAILED: verified arm exceeded its delta — implementation bug, do not report:")
		for _, r := range breached {
			fmt.Fprintf(os.Stderr, "  delta=%.2f  FP/n=%.5f > %.2f\n", r.Delta, r.ErrorRate, r.Delta)
		}
		os.Exit(2)
	}
	fmt.Println("go/no-go: all verified runs within delta ✓")
}

// resultsFile is the top-level results.json shape.
type resultsFile struct {
	Dataset  string      `json:"dataset"`
	N        int         `json:"n"`
	Capacity int         `json:"capacity"` // 0 = unbounded
	Seed     int64       `json:"seed"`
	Workers  int         `json:"workers"`
	NMin     int         `json:"nmin"`
	Runs     []runResult `json:"runs"`
}

// loadJSONL streams the extracted JSONL, keeping at most limit records (0 = all).
func loadJSONL(path string, limit int) ([]record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 8<<20) // lines hold 1024-float vectors

	var records []record
	for sc.Scan() {
		var rec record
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			return nil, fmt.Errorf("parse line %d: %w", len(records)+1, err)
		}
		records = append(records, rec)
		if limit > 0 && len(records) >= limit {
			break
		}
	}
	return records, sc.Err()
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

// writeCSV emits a flat, plotting-friendly table.
func writeCSV(path string, runs []runResult) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()

	fmt.Fprintln(w, "arm,param,delta,nmin,n,tp,fp,exploits,explores,hit_rate,hit_ci_lo,hit_ci_hi,error_rate,error_ci_lo,error_ci_hi,entries_final,within_delta")
	// Stable order: static by threshold, then verified by delta.
	sort.SliceStable(runs, func(i, j int) bool {
		if runs[i].Arm != runs[j].Arm {
			return runs[i].Arm < runs[j].Arm // "static" < "verified"
		}
		return runs[i].Param < runs[j].Param
	})
	for _, r := range runs {
		delta := ""
		nmin := ""
		within := ""
		if r.Arm == "verified" {
			delta = fmt.Sprintf("%.4f", r.Delta)
			nmin = fmt.Sprintf("%d", r.NMin)
			within = fmt.Sprintf("%t", r.WithinDelta)
		}
		fmt.Fprintf(w, "%s,%.4f,%s,%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%s\n",
			r.Arm, r.Param, delta, nmin, r.N, r.TP, r.FP, r.Exploits, r.Explores,
			r.HitRate, r.HitLo, r.HitHi, r.ErrorRate, r.ErrorLo, r.ErrorHi, r.EntriesFinal, within)
	}
	return nil
}

func capacityLabel(capacity int) string {
	if capacity <= 0 {
		return "unbounded"
	}
	return fmt.Sprintf("%d", capacity)
}
