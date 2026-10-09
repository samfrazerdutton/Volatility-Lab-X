import { useState } from 'react'
import { api, ApiRequestError } from '../api/client'
import { useAppState } from '../state/AppState'
import { Metric, MetricGrid, Panel } from './ui'

// Directive section 10/27: incremental vs. independent full rebuild,
// correctness and performance shown separately. Speedup is computed from
// the two measured numbers the server returns, never hardcoded.

export function ComparePanel() {
  const { lastCompare, recordCompare } = useAppState()
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function run() {
    setRunning(true)
    setError(null)
    try {
      recordCompare(await api.compareFullRebuild())
    } catch (e) {
      setError(e instanceof ApiRequestError ? e.body.message ?? e.body.error : String(e))
    } finally {
      setRunning(false)
    }
  }

  return (
    <Panel title="Incremental vs. independent full rebuild">
      <button
        onClick={() => void run()}
        disabled={running}
        style={{
          background: 'var(--accent-dim)',
          border: '1px solid var(--accent)',
          color: 'var(--text)',
          padding: '6px 14px',
          borderRadius: 3,
          fontSize: 12,
          marginBottom: 12,
        }}
      >
        {running ? 'Running…' : 'Run comparison'}
      </button>
      {error && <div style={{ color: 'var(--bad)', fontSize: 11, marginBottom: 8 }}>{error}</div>}

      {lastCompare && (
        <>
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase', marginBottom: 6 }}>
              Correctness (checked independently of timing)
            </div>
            <MetricGrid>
              <Metric
                label="PnL difference"
                value={`$${lastCompare.pnl_diff.toFixed(9)}`}
                tone={lastCompare.pnl_diff < 1e-6 ? 'good' : 'bad'}
              />
              <Metric
                label="Max surface vol diff"
                value={lastCompare.max_vol_diff.toExponential(2)}
                tone={lastCompare.max_vol_diff < 1e-9 ? 'good' : 'bad'}
              />
            </MetricGrid>
          </div>

          <div>
            <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase', marginBottom: 6 }}>
              Performance (measured separately from correctness)
            </div>
            <MetricGrid>
              <Metric
                label="Incremental (last update)"
                value={lastCompare.incremental_measured ? `${lastCompare.incremental_us.toFixed(1)} µs` : 'n/a'}
              />
              <Metric label="Full rebuild (just now)" value={`${lastCompare.full_rebuild_ms.toFixed(3)} ms`} />
              <Metric label="Speedup" value={lastCompare.speedup ? `${lastCompare.speedup.toFixed(2)}x` : 'n/a'} />
              <Metric
                label="Quotes examined"
                value={`${lastCompare.quotes_examined} / ${lastCompare.quotes_total}`}
              />
              <Metric label="Nodes recomputed" value={`${lastCompare.nodes_recomputed} / ${lastCompare.nodes_total}`} />
            </MetricGrid>
            {!lastCompare.incremental_measured && (
              <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 6 }}>
                No incremental update has run yet this session -- apply a quote edit first to get a
                real incremental timing to compare against.
              </div>
            )}
          </div>

          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase', marginBottom: 6 }}>
              Per-expiry surface vol, both paths
            </div>
            <table className="mono" style={{ fontSize: 11, width: '100%' }}>
              <thead>
                <tr style={{ color: 'var(--text-faint)', textAlign: 'left' }}>
                  <th>T</th>
                  <th>Incremental</th>
                  <th>Full rebuild</th>
                  <th>Diff</th>
                </tr>
              </thead>
              <tbody>
                {lastCompare.surface_diffs.map((d) => (
                  <tr key={d.years}>
                    <td>{d.years.toFixed(4)}</td>
                    <td>{(d.incremental_vol * 100).toFixed(4)}%</td>
                    <td>{(d.full_rebuild_vol * 100).toFixed(4)}%</td>
                    <td style={{ color: d.diff < 1e-9 ? 'var(--good)' : 'var(--bad)' }}>{d.diff.toExponential(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  )
}
