import { useState } from 'react'
import { api, ApiRequestError } from '../api/client'
import type { Quote } from '../api/types'
import { useAppState } from '../state/AppState'
import { Panel } from './ui'

// The interactive market-event editor, directive section 8. Every Apply
// goes through the real C++ service: apply_event -> recompute() -> the
// displayed numbers are the engine's own returned result, never a
// frontend-mutated copy of the quote.

export function QuoteEditorPanel({ quote, onApplied }: { quote: Quote; onApplied: () => void }) {
  const { recordApply } = useAppState()
  const [bid, setBid] = useState(quote.bid)
  const [ask, setAsk] = useState(quote.ask)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const bidChanged = bid !== quote.bid
  const askChanged = ask !== quote.ask
  const valid = bid >= 0 && ask >= 0 && bid <= ask

  async function apply() {
    setSubmitting(true)
    setError(null)
    try {
      const result = await api.applyEvent({
        years: quote.years,
        strike: quote.strike,
        option_type: quote.type,
        bid,
        ask,
      })
      recordApply(result)
      onApplied()
    } catch (e) {
      setError(e instanceof ApiRequestError ? e.body.message ?? e.body.error : String(e))
    } finally {
      setSubmitting(false)
    }
  }

  function cancel() {
    setBid(quote.bid)
    setAsk(quote.ask)
    setError(null)
  }

  return (
    <Panel title={`Edit quote -- T=${quote.years.toFixed(4)} K=${quote.strike} ${quote.type}`}>
      <table className="mono" style={{ fontSize: 12, width: '100%' }}>
        <thead>
          <tr style={{ color: 'var(--text-faint)', textAlign: 'left' }}>
            <th style={{ paddingBottom: 6 }}>Field</th>
            <th>Original</th>
            <th>Proposed</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ color: 'var(--text-dim)' }}>Bid</td>
            <td>{quote.bid.toFixed(4)}</td>
            <td>
              <input
                type="number"
                step="0.01"
                value={bid}
                onChange={(e) => setBid(Number(e.target.value))}
                style={{
                  ...inputStyle,
                  borderColor: bidChanged ? 'var(--accent)' : 'var(--border-strong)',
                  color: bidChanged ? 'var(--accent)' : 'var(--text)',
                }}
              />
            </td>
          </tr>
          <tr>
            <td style={{ color: 'var(--text-dim)' }}>Ask</td>
            <td>{quote.ask.toFixed(4)}</td>
            <td>
              <input
                type="number"
                step="0.01"
                value={ask}
                onChange={(e) => setAsk(Number(e.target.value))}
                style={{
                  ...inputStyle,
                  borderColor: askChanged ? 'var(--accent)' : 'var(--border-strong)',
                  color: askChanged ? 'var(--accent)' : 'var(--text)',
                }}
              />
            </td>
          </tr>
        </tbody>
      </table>

      {!valid && (
        <div style={{ color: 'var(--bad)', fontSize: 11, marginTop: 8 }}>
          Invalid: bid and ask must be non-negative, and bid must not exceed ask.
        </div>
      )}
      {error && <div style={{ color: 'var(--bad)', fontSize: 11, marginTop: 8 }}>Server rejected update: {error}</div>}

      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        <button
          disabled={!valid || submitting || (!bidChanged && !askChanged)}
          onClick={() => void apply()}
          style={{
            background: 'var(--accent-dim)',
            border: '1px solid var(--accent)',
            color: 'var(--text)',
            padding: '6px 14px',
            borderRadius: 3,
            fontSize: 12,
            opacity: !valid || submitting ? 0.5 : 1,
          }}
        >
          {submitting ? 'Applying…' : 'Apply update'}
        </button>
        <button
          onClick={cancel}
          disabled={submitting}
          style={{
            background: 'transparent',
            border: '1px solid var(--border-strong)',
            color: 'var(--text-dim)',
            padding: '6px 14px',
            borderRadius: 3,
            fontSize: 12,
          }}
        >
          Cancel
        </button>
      </div>
    </Panel>
  )
}

const inputStyle: React.CSSProperties = {
  background: 'var(--surface-raised)',
  border: '1px solid var(--border-strong)',
  color: 'var(--text)',
  padding: '4px 6px',
  borderRadius: 3,
  width: 90,
  fontFamily: 'var(--mono)',
  fontSize: 12,
}
