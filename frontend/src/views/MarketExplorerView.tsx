import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { Greeks, Quote } from '../api/types'
import { ComparePanel } from '../components/ComparePanel'
import { QuoteEditorPanel } from '../components/QuoteEditorPanel'
import { EmptyState, Metric, MetricGrid, Panel, StatusPill } from '../components/ui'
import { useAppState } from '../state/AppState'

type SortKey = keyof Pick<Quote, 'years' | 'strike' | 'bid' | 'ask' | 'mid' | 'implied_vol' | 'volume' | 'open_interest'>

export function MarketExplorerView() {
  const { health, expiries, refreshExpiries, selectedQuote, setSelectedQuote } = useAppState()
  const [quotes, setQuotes] = useState<Quote[]>([])
  const [matched, setMatched] = useState(0)
  const [expiryFilter, setExpiryFilter] = useState<number | 'all'>('all')
  const [typeFilter, setTypeFilter] = useState<'all' | 'call' | 'put'>('all')
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('strike')
  const [sortDir, setSortDir] = useState<1 | -1>(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [greeks, setGreeks] = useState<Greeks | null>(null)

  useEffect(() => {
    void refreshExpiries()
  }, [refreshExpiries])

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const res = await api.quotes({
        expiry: expiryFilter === 'all' ? undefined : expiryFilter,
        type: typeFilter === 'all' ? undefined : typeFilter,
        limit: 1000,
      })
      setQuotes(res.quotes)
      setMatched(res.matched)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expiryFilter, typeFilter])

  useEffect(() => {
    if (!selectedQuote) {
      setGreeks(null)
      return
    }
    api
      .quoteGreeks({
        years: selectedQuote.years,
        strike: selectedQuote.strike,
        type: selectedQuote.type,
        vol: selectedQuote.implied_vol,
      })
      .then(setGreeks)
      .catch(() => setGreeks(null))
  }, [selectedQuote])

  const filtered = useMemo(() => {
    const s = search.trim()
    let rows = quotes
    if (s) {
      rows = rows.filter((q) => String(q.strike).includes(s))
    }
    return rows
      .slice()
      .sort((a, b) => (a[sortKey] - b[sortKey]) * sortDir)
  }, [quotes, search, sortKey, sortDir])

  function toggleSort(key: SortKey) {
    if (key === sortKey) setSortDir((d) => (d === 1 ? -1 : 1))
    else {
      setSortKey(key)
      setSortDir(1)
    }
  }

  if (!health?.market_loaded) {
    return <EmptyState message="No market loaded. Use 'Load sample market' in the top bar." />
  }

  return (
    <div style={{ display: 'flex', gap: 16, padding: 20, height: '100%', minHeight: 0 }}>
      <div style={{ flex: 2, display: 'flex', flexDirection: 'column', gap: 12, minWidth: 0 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <select className="mono" value={expiryFilter} onChange={(e) => setExpiryFilter(e.target.value === 'all' ? 'all' : Number(e.target.value))} style={selectStyle}>
            <option value="all">All expiries</option>
            {expiries.map((e) => (
              <option key={e.years} value={e.years}>
                T={e.years.toFixed(4)} ({e.quote_count}q)
              </option>
            ))}
          </select>
          <select className="mono" value={typeFilter} onChange={(e) => setTypeFilter(e.target.value as typeof typeFilter)} style={selectStyle}>
            <option value="all">Call + Put</option>
            <option value="call">Call</option>
            <option value="put">Put</option>
          </select>
          <input
            className="mono"
            placeholder="Search strike…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ ...selectStyle, width: 140 }}
          />
          <span style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            {filtered.length} shown / {matched} matched
          </span>
        </div>

        <Panel title="Quotes">
          {loading && <EmptyState message="Loading…" />}
          {error && <EmptyState message={`Error: ${error}`} />}
          {!loading && !error && filtered.length === 0 && <EmptyState message="No quotes match the current filters." />}
          {!loading && !error && filtered.length > 0 && (
            <div style={{ maxHeight: 520, overflow: 'auto' }}>
              <table className="mono" style={{ fontSize: 11, width: '100%' }}>
                <thead style={{ position: 'sticky', top: 0, background: 'var(--surface-raised)' }}>
                  <tr style={{ textAlign: 'right', color: 'var(--text-faint)' }}>
                    <Th label="T" onClick={() => toggleSort('years')} align="left" />
                    <Th label="Strike" onClick={() => toggleSort('strike')} />
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>Type</th>
                    <Th label="Bid" onClick={() => toggleSort('bid')} />
                    <Th label="Ask" onClick={() => toggleSort('ask')} />
                    <Th label="Mid" onClick={() => toggleSort('mid')} />
                    <Th label="IV" onClick={() => toggleSort('implied_vol')} />
                    <Th label="Vol" onClick={() => toggleSort('volume')} />
                    <Th label="OI" onClick={() => toggleSort('open_interest')} />
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((q, i) => {
                    const isSelected =
                      selectedQuote && selectedQuote.strike === q.strike && selectedQuote.years === q.years && selectedQuote.type === q.type
                    return (
                      <tr
                        key={i}
                        onClick={() => setSelectedQuote(q)}
                        style={{
                          textAlign: 'right',
                          cursor: 'pointer',
                          background: isSelected ? 'var(--accent-dim)' : i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)',
                        }}
                      >
                        <td style={{ textAlign: 'left', padding: '4px 8px' }}>{q.years.toFixed(4)}</td>
                        <td style={{ padding: '4px 8px' }}>{q.strike.toFixed(2)}</td>
                        <td style={{ textAlign: 'left', padding: '4px 8px', color: q.type === 'call' ? 'var(--good)' : 'var(--bad)' }}>
                          {q.type}
                        </td>
                        <td style={{ padding: '4px 8px' }}>{q.bid.toFixed(3)}</td>
                        <td style={{ padding: '4px 8px' }}>{q.ask.toFixed(3)}</td>
                        <td style={{ padding: '4px 8px' }}>{q.mid.toFixed(3)}</td>
                        <td style={{ padding: '4px 8px' }}>{(q.implied_vol * 100).toFixed(2)}%</td>
                        <td style={{ padding: '4px 8px' }}>{q.volume.toFixed(0)}</td>
                        <td style={{ padding: '4px 8px' }}>{q.open_interest.toFixed(0)}</td>
                        <td style={{ textAlign: 'left', padding: '4px 8px' }}>
                          <StatusPill
                            text={q.status}
                            tone={q.status === 'ok' ? 'good' : q.status === 'rejected' ? 'bad' : q.status === 'degraded' ? 'warn' : 'neutral'}
                          />
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <ComparePanel />
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12, minWidth: 320 }}>
        {selectedQuote ? (
          <>
            <Panel title="Selected instrument">
              <MetricGrid>
                <Metric label="Expiry" value={selectedQuote.years.toFixed(4)} />
                <Metric label="Strike" value={selectedQuote.strike} />
                <Metric label="Type" value={selectedQuote.type} />
                <Metric label="Implied vol" value={`${(selectedQuote.implied_vol * 100).toFixed(3)}%`} />
                <Metric label="Volume" value={selectedQuote.volume} />
                <Metric label="Open interest" value={selectedQuote.open_interest} />
              </MetricGrid>
              {greeks && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase', marginBottom: 6 }}>
                    Greeks (computed via black_scholes_greeks at this quote&apos;s own implied vol)
                  </div>
                  <MetricGrid>
                    <Metric label="Delta" value={greeks.delta.toFixed(4)} />
                    <Metric label="Gamma" value={greeks.gamma.toFixed(5)} />
                    <Metric label="Vega" value={greeks.vega.toFixed(4)} />
                    <Metric label="Theta" value={greeks.theta.toFixed(4)} />
                  </MetricGrid>
                </div>
              )}
            </Panel>
            <QuoteEditorPanel quote={selectedQuote} onApplied={() => void load()} />
          </>
        ) : (
          <Panel title="Selected instrument">
            <EmptyState message="Select a row to inspect and edit it." />
          </Panel>
        )}
      </div>
    </div>
  )
}

function Th({ label, onClick, align = 'right' }: { label: string; onClick: () => void; align?: 'left' | 'right' }) {
  return (
    <th onClick={onClick} style={{ textAlign: align, padding: '6px 8px', cursor: 'pointer', userSelect: 'none' }}>
      {label}
    </th>
  )
}

const selectStyle: React.CSSProperties = {
  background: 'var(--surface-raised)',
  border: '1px solid var(--border-strong)',
  color: 'var(--text)',
  fontSize: 12,
  padding: '5px 8px',
  borderRadius: 3,
}
