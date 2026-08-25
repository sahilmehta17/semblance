package judge

import (
	"context"
	"log/slog"
	"sync"
)

// Job is one async labeling task. When labeling completes, OnResult is called
// with the verdict (c). OnResult runs on a worker goroutine, so it must be
// safe to call off the request path — in the gateway it appends the observation
// and does the guarded insert, both of which the cache serializes internally.
type Job struct {
	Reference string
	Candidate string
	OnResult  func(equivalent bool)
}

// Labeler runs judging asynchronously behind a bounded queue and a pool of
// workers. Submit never blocks: if the queue is full the job is dropped and
// counted, so a backlog can never stall request handling.
type Labeler struct {
	judge   Judge
	queue   chan Job
	logger  *slog.Logger
	wg      sync.WaitGroup
	dropped atomicCounter
}

// NewLabeler starts a Labeler with the given judge, queue size, and worker
// count. Call Close to drain and stop it.
func NewLabeler(j Judge, queueSize, workers int, logger *slog.Logger) *Labeler {
	if queueSize < 1 {
		queueSize = 1
	}
	if workers < 1 {
		workers = 1
	}
	if logger == nil {
		logger = slog.Default()
	}
	l := &Labeler{
		judge:  j,
		queue:  make(chan Job, queueSize),
		logger: logger,
	}
	l.wg.Add(workers)
	for i := 0; i < workers; i++ {
		go l.worker()
	}
	return l
}

func (l *Labeler) worker() {
	defer l.wg.Done()
	for job := range l.queue {
		// A judge error is treated as "not equivalent" (c=0): conservative,
		// since it can only cause an extra cache entry, never a wrong hit.
		eq, err := l.judge.Equivalent(context.Background(), job.Reference, job.Candidate)
		if err != nil {
			l.logger.Debug("judge error; labeling not-equivalent", "err", err)
			eq = false
		}
		if job.OnResult != nil {
			job.OnResult(eq)
		}
	}
}

// Submit enqueues a job without blocking. It returns false if the queue was
// full (the job is dropped). The non-blocking send is the classic Go idiom: a
// select with a default case.
func (l *Labeler) Submit(job Job) bool {
	select {
	case l.queue <- job:
		return true
	default:
		l.dropped.inc()
		l.logger.Debug("labeling queue full; dropping job", "dropped_total", l.dropped.get())
		return false
	}
}

// Dropped returns how many jobs have been dropped due to a full queue.
func (l *Labeler) Dropped() int64 { return l.dropped.get() }

// QueueDepth returns the number of jobs currently waiting in the queue. Safe to
// call concurrently (len on a channel is atomic).
func (l *Labeler) QueueDepth() int { return len(l.queue) }

// Close stops accepting new jobs and waits for in-flight jobs to finish. After
// Close, Submit will panic (send on closed channel) — callers must stop
// submitting first, which the gateway does during graceful shutdown.
func (l *Labeler) Close() {
	close(l.queue)
	l.wg.Wait()
}

// atomicCounter is a tiny lock-free counter (kept local to avoid pulling in
// sync/atomic types across the package surface).
type atomicCounter struct {
	mu sync.Mutex
	n  int64
}

func (c *atomicCounter) inc() {
	c.mu.Lock()
	c.n++
	c.mu.Unlock()
}

func (c *atomicCounter) get() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.n
}
