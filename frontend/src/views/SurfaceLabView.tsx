import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { SmileChart } from '../components/SmileChart'
import { EmptyState, Metric, MetricGrid, Panel, StatusPill } from '../components/ui'
import { useAppState } from '../state/AppState'
import type { SurfacePoint, SurfaceResponse } from '../api/types'

// The 2D smile view + numerical detail panel the directive's section 6
// asks for. The 3D WebGL implied-vol surface (that section's flagship
// requirement) is NOT implemented in this build -- it is a materially
// larger piece of work (camera controls, a mesh built from the surface's
// own batch query API, a legend-mapped colour scale) than this vertical
// slice's scope. This view is the honest subset: real calibrated-vs-
// observed data, for one expiry at a time, selectable and inspectable.

export function SurfaceLabView() {
  const { expiries, refreshExpiries, health } = useAppState()
  const [years, setYears] = useState<number | null>(null)
  const [surface, setSurface] = useState<SurfaceResponse | null>(null)
  const [selectedPoint, setSelectedPoint] = useState<SurfacePoint | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void refreshExpiries()
  }, [refreshExpiries])

  // Derived, not stored: the effective selection is the user's explicit
  // choice if they made one, otherwise the first available expiry --
  // computed during render rather than synchronised via a setState effect.
  const effectiveYears = years ?? (expiries.length > 0 ? expiries[0].years : null)

  useEffect(() => {
    if (effectiveYears === null) return
    setLoading(true)
    setError(null)
    setSelectedPoint(null)
    api
      .surface(effectiveYears)
      .then(setSurface)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [effectiveYears])

  if (!health?.market_loaded) {
    return <EmptyState message="No market loaded. Use 'Load sample market' in the top bar." />
  }

  return (
    <div style={{ display: 'flex', gap: 16, padding: 20, height: '100%' }}>
      <div style={{ width: 160, flexShrink: 0 }}>
        <Panel title="Expiries">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {expiries.map((e) => (
              <button
                key={e.years}
                onClick={() => setYears(e.years)}
                className="mono"
                style={{
                  textAlign: 'left',
                  background: effectiveYears === e.years ? 'var(--accent-dim)' : 'transparent',
                  border: '1px solid',
                  borderColor: effectiveYears === e.years ? 'var(--accent)' : 'var(--border)',
                  color: 'var(--text)',
                  fontSize: 12,
                  padding: '6px 8px',
                  borderRadius: 3,
                  marginBottom: 2,
                }}
              >
                T={e.years.toFixed(4)} ({e.quote_count}q)
              </button>
            ))}
          </div>
        </Panel>
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
        <Panel title={`Smile -- T=${effectiveYears?.toFixed(4) ?? ''} (observed vs. calibrated SVI)`}>
          {loading && <EmptyState message="Loading…" />}
          {error && <EmptyState message={`Error: ${error}`} />}
          {surface && !loading && (
            <SmileChart
              observed={surface.observed}
              calibrated={surface.calibrated}
              onSelect={setSelectedPoint}
              selected={selectedPoint}
            />
          )}
        </Panel>

        {surface && (
          <Panel
            title="Calibration diagnostics"
            right={<StatusPill text={surface.fit.status} tone={surface.fit.ok ? 'good' : 'bad'} />}
          >
            <MetricGrid>
              <Metric label="RMS vol error" value={`${(surface.fit.rms_vol_error * 100).toFixed(3)}%`} />
              <Metric label="Max vol error" value={`${(surface.fit.max_vol_error * 100).toFixed(3)}%`} />
              <Metric label="Quotes used" value={`${surface.fit.quotes_used} / ${surface.fit.quotes_available}`} />
              <Metric label="Inner solves" value={surface.fit.inner_solves} />
              <Metric label="Outer iterations" value={surface.fit.outer_iterations} />
              <Metric label="Active constraints" value={surface.fit.active_constraints} />
            </MetricGrid>
            <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 8 }}>
              Recomputed on demand from this expiry&apos;s own quotes via the same
              calibrate_svi_slice the engine itself calls -- a real recalculation for display, not a
              cached value that could drift from what&apos;s shown in the chart above.
            </div>
          </Panel>
        )}

        {selectedPoint && (
          <Panel title="Selected point">
            <MetricGrid>
              <Metric label="Type" value={selectedPoint.type} />
              <Metric label="Strike" value={selectedPoint.strike} />
              <Metric label="Bid" value={selectedPoint.bid} />
              <Metric label="Ask" value={selectedPoint.ask} />
              <Metric label="Mid" value={selectedPoint.mid} />
              <Metric label="Implied vol" value={`${(selectedPoint.implied_vol * 100).toFixed(3)}%`} />
              <Metric label="Status" value={selectedPoint.status} />
            </MetricGrid>
            <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 8 }}>
              This is an observed market quote, not a model estimate. To edit it, go to Market
              Explorer.
            </div>
          </Panel>
        )}
      </div>
    </div>
  )
}
