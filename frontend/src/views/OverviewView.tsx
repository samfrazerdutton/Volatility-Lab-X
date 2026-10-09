import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { EmptyState, Metric, MetricGrid, Panel } from '../components/ui'
import { useAppState } from '../state/AppState'
import type { QuotesResponse } from '../api/types'

// Answers the four questions the directive's Overview section asks for:
// what is loaded, what has the engine calculated, is it numerically valid,
// how much work did the last computation require -- every value sourced
// from a real endpoint, never invented when a metric isn't exposed yet.

export function OverviewView() {
  const { health, expiries, events, lastApply, lastCompare, refreshHealth, refreshExpiries, refreshEvents } =
    useAppState()
  const [quoteSummary, setQuoteSummary] = useState<QuotesResponse | null>(null)

  useEffect(() => {
    void refreshHealth()
    void refreshExpiries()
    void refreshEvents()
    void api.quotes({ limit: 1 }).then(setQuoteSummary)
  }, [refreshHealth, refreshExpiries, refreshEvents])

  if (!health?.market_loaded) {
    return <EmptyState message="No market loaded. Use 'Load sample market' in the top bar." />
  }

  const totalQuotes = quoteSummary?.matched ?? 0
  const expiryCount = expiries.length

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: 20 }}>
      <Panel title="What data is loaded">
        <MetricGrid>
          <Metric label="Quotes" value={totalQuotes} definition="OptionQuote count in the engine's current book" />
          <Metric label="Expiries" value={expiryCount} definition="Distinct ExpirySlice nodes" />
          <Metric label="Engine" value={health.market_loaded ? 'ready' : 'empty'} tone={health.market_loaded ? 'good' : 'bad'} />
        </MetricGrid>
      </Panel>

      <Panel title="What the engine last calculated">
        {lastApply ? (
          <MetricGrid>
            <Metric
              label="Nodes recomputed"
              value={`${lastApply.report.recomputed_nodes} / ${lastApply.report.total_nodes}`}
              definition="RecomputeReport.recomputed_nodes / total_nodes"
            />
            <Metric
              label="Calibrations run"
              value={lastApply.report.calibrations_run}
              definition="ExpirySlice nodes actually recalibrated this update"
            />
            <Metric
              label="Quotes examined"
              value={`${lastApply.report.quotes_examined} / ${lastApply.report.quotes_total}`}
              definition="Quotes read while recomputing dirty ExpirySlice nodes vs. total book size"
            />
            <Metric
              label="Last update duration"
              value={`${lastApply.report.latency_us.toFixed(1)} µs`}
              definition="Wall-clock time of the last apply_event + recompute() pass"
            />
          </MetricGrid>
        ) : (
          <EmptyState message="No market event applied yet -- see Market Explorer." />
        )}
      </Panel>

      <Panel title="Is the result numerically valid">
        {lastCompare ? (
          <MetricGrid>
            <Metric
              label="PnL diff vs full rebuild"
              value={`$${lastCompare.pnl_diff.toFixed(9)}`}
              tone={lastCompare.pnl_diff < 1e-6 ? 'good' : 'bad'}
              definition="abs(incremental PnL - independent full_rebuild() PnL)"
            />
            <Metric
              label="Max surface vol diff"
              value={lastCompare.max_vol_diff.toExponential(2)}
              tone={lastCompare.max_vol_diff < 1e-9 ? 'good' : 'bad'}
              definition="Largest |incremental vol - full rebuild vol| across expiries"
            />
            <Metric
              label="Measured speedup"
              value={lastCompare.speedup ? `${lastCompare.speedup.toFixed(2)}x` : 'n/a'}
              definition="full_rebuild_ms / incremental latency, both measured"
            />
          </MetricGrid>
        ) : (
          <EmptyState message="No full-rebuild comparison run yet -- see Market Explorer." />
        )}
      </Panel>

      {lastApply && (
        <Panel title="State hash">
          <div className="mono" style={{ fontSize: 13, color: 'var(--accent)' }}>
            {lastApply.state_hash}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 4 }}>
            FNV-1a fingerprint of the engine&apos;s current quote book (years, strike, type, mid) --
            a display-only fingerprint, distinct from the replay-stream hash used by the C++ test
            suite&apos;s own determinism checks.
          </div>
        </Panel>
      )}

      <Panel title="Recent experiment timeline">
        {events.length === 0 ? (
          <EmptyState message="No events yet." />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {events
              .slice()
              .reverse()
              .map((e, i) => (
                <div key={i} style={{ display: 'flex', gap: 10, fontSize: 12, alignItems: 'baseline' }}>
                  <span className="mono" style={{ color: 'var(--text-faint)', width: 150, flexShrink: 0 }}>
                    {new Date(e.timestamp_ms).toLocaleTimeString()}
                  </span>
                  <span style={{ color: 'var(--accent)', width: 150, flexShrink: 0 }}>{e.type}</span>
                  <span style={{ color: 'var(--text-dim)' }}>{e.description}</span>
                </div>
              ))}
          </div>
        )}
      </Panel>
    </div>
  )
}
