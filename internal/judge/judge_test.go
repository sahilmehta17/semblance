package judge

import (
	"context"
	"io"
	"log/slog"
	"strings"
	"testing"
	"time"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestExactJudge(t *testing.T) {
	j := ExactJudge{}
	if eq, _ := j.Equivalent(context.Background(), "paris", " paris "); !eq {
		t.Error("exact judge should ignore surrounding whitespace")
	}
	if eq, _ := j.Equivalent(context.Background(), "paris", "london"); eq {
		t.Error("different responses must not be equivalent")
	}
}

// stubLLM always returns the same verdict, to prove the long-response path
// delegates to it.
type stubLLM struct{ verdict bool }

func (s stubLLM) Equivalent(_ context.Context, _, _ string) (bool, error) { return s.verdict, nil }

func TestDefaultJudgeShortUsesExact(t *testing.T) {
	// No LLM configured; short responses use exact match.
	d := NewDefaultJudge(nil)
	if eq, _ := d.Equivalent(context.Background(), "yes", "yes"); !eq {
		t.Error("short equal responses should be equivalent")
	}
	if eq, _ := d.Equivalent(context.Background(), "yes", "no"); eq {
		t.Error("short unequal responses should not be equivalent")
	}
}

func TestDefaultJudgeLongDelegatesToLLM(t *testing.T) {
	long := strings.Repeat("x", 300) // exceeds ShortLen
	// LLM says "equivalent" even though the strings differ; proves delegation.
	d := NewDefaultJudge(stubLLM{verdict: true})
	if eq, _ := d.Equivalent(context.Background(), long, long+"different"); !eq {
		t.Error("long responses should be judged by the LLM path")
	}

	// With no LLM, the long path falls back to exact match (conservative).
	d2 := NewDefaultJudge(nil)
	if eq, _ := d2.Equivalent(context.Background(), long, long+"different"); eq {
		t.Error("fallback should use exact match and report not-equivalent")
	}
}

func TestLabelerAsyncResult(t *testing.T) {
	l := NewLabeler(ExactJudge{}, 8, 2, discardLogger())
	defer l.Close()

	got := make(chan bool, 1)
	if !l.Submit(Job{Reference: "paris", Candidate: "paris", OnResult: func(eq bool) { got <- eq }}) {
		t.Fatal("submit should succeed")
	}
	select {
	case eq := <-got:
		if !eq {
			t.Error("expected equivalent=true")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("labeling did not complete in time")
	}
}

// blockingJudge blocks each call until release is closed, and signals when a
// call has started — enough control to force a full queue deterministically.
type blockingJudge struct {
	started chan struct{}
	release chan struct{}
}

func (b *blockingJudge) Equivalent(_ context.Context, _, _ string) (bool, error) {
	b.started <- struct{}{}
	<-b.release
	return true, nil
}

func TestLabelerBoundedQueueDropsWhenFull(t *testing.T) {
	bj := &blockingJudge{started: make(chan struct{}, 10), release: make(chan struct{})}
	l := NewLabeler(bj, 1, 1, discardLogger()) // queue size 1, single worker

	noop := func(bool) {}
	if !l.Submit(Job{OnResult: noop}) { // job1: worker will pick this up
		t.Fatal("submit 1 should succeed")
	}
	<-bj.started // worker is now busy on job1, queue is empty

	if !l.Submit(Job{OnResult: noop}) { // job2: fills the queue (size 1)
		t.Fatal("submit 2 should succeed (queue has room)")
	}
	if l.Submit(Job{OnResult: noop}) { // job3: queue full -> dropped
		t.Error("submit 3 should be dropped (queue full)")
	}

	close(bj.release) // let the worker drain
	l.Close()

	if l.Dropped() != 1 {
		t.Errorf("Dropped() = %d, want 1", l.Dropped())
	}
}
