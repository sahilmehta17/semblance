// Package metrics holds the Prometheus collectors for the gateway and the
// /metrics handler. Everything lives on a private registry so tests can build
// isolated instances and the default global registry stays untouched.
//
// Honesty note (see DECISIONS.md): there is deliberately NO live
// error_rate_realized metric. In production you cannot compute the true FP/n —
// on an exploit you serve the cached answer and never call the model, so you
// never learn whether it was wrong. Realized error is a benchmark-only quantity
// (it needs ground truth). What we expose live is the configured delta target
// plus honestly-labeled OBSERVABLE proxies: the judge-observed c=0 rate among
// explores, and the hit rate (derivable from requests_total). The whole value
// of the guarantee is that it bounds an error you cannot otherwise observe live.
package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const namespace = "semblance"

// Metrics bundles every collector plus the registry they live on.
type Metrics struct {
	reg *prometheus.Registry

	RequestsTotal *prometheus.CounterVec // outcome=exploit|explore|bypass
	BypassTotal   *prometheus.CounterVec // reason
	CacheEntries  prometheus.Gauge       // set via RegisterCacheEntriesFunc if desired
	EntryObs      prometheus.Histogram
	Tau           prometheus.Histogram
	Similarity    prometheus.Histogram
	RefitSeconds  prometheus.Histogram
	JudgeQueue    prometheus.Gauge
	BackendTokens *prometheus.CounterVec // direction=prompt|completion
	EmbedTokens   prometheus.Counter
	CostUSD       *prometheus.CounterVec // kind=spent|embedding|saved
	BudgetCents   *prometheus.GaugeVec   // key
	DeltaTarget   prometheus.Gauge       // configured error budget (the target)
	ExploresLabel prometheus.Counter     // explores that got a ground-truth-ish label
	ExploresC0    prometheus.Counter     // of those, judge said c=0 (would-be wrong)
}

// New builds and registers all collectors on a fresh registry.
func New() *Metrics {
	reg := prometheus.NewRegistry()
	m := &Metrics{
		reg: reg,
		RequestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace, Name: "requests_total",
			Help: "Chat-completion requests by cache outcome.",
		}, []string{"outcome"}),
		BypassTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace, Name: "bypass_total",
			Help: "Requests that bypassed the cache, by reason.",
		}, []string{"reason"}),
		CacheEntries: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace, Name: "cache_entries",
			Help: "Current number of cache entries.",
		}),
		EntryObs: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace, Name: "entry_observations",
			Help:    "Observation-set size of the matched entry at decision time.",
			Buckets: prometheus.ExponentialBuckets(1, 2, 12), // 1..2048
		}),
		Tau: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace, Name: "tau",
			Help:    "Exploration probability tau at decision time.",
			Buckets: prometheus.LinearBuckets(0, 0.1, 11), // 0..1
		}),
		Similarity: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace, Name: "similarity",
			Help:    "Cosine similarity to nearest neighbor (hits and misses).",
			Buckets: prometheus.LinearBuckets(0, 0.05, 21), // 0..1
		}),
		RefitSeconds: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace, Name: "policy_refit_seconds",
			Help:    "Time to fit the logistic model and compute tau.",
			Buckets: prometheus.ExponentialBuckets(1e-5, 3, 10), // 10us..
		}),
		JudgeQueue: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace, Name: "judge_queue_depth",
			Help: "Pending jobs in the async labeling queue.",
		}),
		BackendTokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace, Name: "backend_tokens_total",
			Help: "Tokens billed by the backend, by direction.",
		}, []string{"direction"}),
		EmbedTokens: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace, Name: "embed_tokens_total",
			Help: "Tokens billed by the embedding provider.",
		}),
		CostUSD: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace, Name: "cost_usd_total",
			Help: "USD cost by kind: spent (explores), embedding, saved (modeled, on hits).",
		}, []string{"kind"}),
		BudgetCents: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace, Name: "budget_remaining_cents",
			Help: "Remaining per-key budget in cents.",
		}, []string{"key"}),
		DeltaTarget: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace, Name: "policy_delta_target",
			Help: "Configured error budget delta (the TARGET; realized error is not observable live).",
		}),
		ExploresLabel: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace, Name: "explores_labeled_total",
			Help: "Explores that received an equivalence label from the judge.",
		}),
		ExploresC0: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace, Name: "explores_incorrect_total",
			Help: "Labeled explores where the neighbor's answer would have been wrong (c=0). Proxy, not FP/n.",
		}),
	}
	reg.MustRegister(
		m.RequestsTotal, m.BypassTotal, m.CacheEntries, m.EntryObs, m.Tau,
		m.Similarity, m.RefitSeconds, m.JudgeQueue, m.BackendTokens, m.EmbedTokens,
		m.CostUSD, m.BudgetCents, m.DeltaTarget, m.ExploresLabel, m.ExploresC0,
	)
	return m
}

// Handler serves the metrics in Prometheus text format.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.reg, promhttp.HandlerOpts{})
}

// RegisterCacheEntriesFunc registers a GaugeFunc for cache_entries evaluated on
// each scrape (so it is correct even when the gateway is idle). Replaces the
// plain CacheEntries gauge; call once at startup.
func (m *Metrics) RegisterCacheEntriesFunc(fn func() float64) {
	m.reg.Unregister(m.CacheEntries)
	m.reg.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: namespace, Name: "cache_entries",
		Help: "Current number of cache entries.",
	}, fn))
}

// RegisterJudgeQueueFunc registers a GaugeFunc for judge_queue_depth.
func (m *Metrics) RegisterJudgeQueueFunc(fn func() float64) {
	m.reg.Unregister(m.JudgeQueue)
	m.reg.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: namespace, Name: "judge_queue_depth",
		Help: "Pending jobs in the async labeling queue.",
	}, fn))
}
