import { useEffect, useState } from 'react'
import { useAppState } from '../state/AppState'

const REGIMES = ['normal', 'high-vol', 'crash', 'vol-crush', 'earnings', 'illiquid']

export function TopBar() {
  const { health, loading, reloadMarket, lastError } = useAppState()
  const [regime, setRegime] = useState('normal')

  useEffect(() => {
    // Health is polled lazily by views that need it; this just gets an
    // initial read so the bar isn't blank on first paint.
  }, [])

  return (
    <div style={styles.bar}>
      <div style={styles.left}>
        <span style={styles.wordmark}>VOLATILITY-LAB-X</span>
        <span style={styles.subtitle}>Quant Compute Workbench</span>
      </div>

      <div style={styles.center}>
        <StatusDot ok={!!health?.market_loaded} label={health?.market_loaded ? 'market loaded' : 'no market'} />
        <span className="mono" style={styles.meta}>
          {health ? `${health.compiler} ${health.compiler_version} · ${health.build_type} · ${health.simd}` : 'connecting…'}
        </span>
      </div>

      <div style={styles.right}>
        <select
          className="mono"
          value={regime}
          onChange={(e) => setRegime(e.target.value)}
          style={styles.select}
          disabled={loading}
        >
          {REGIMES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
        <button style={styles.loadBtn} disabled={loading} onClick={() => void reloadMarket(regime)}>
          {loading ? 'Loading…' : 'Load sample market'}
        </button>
      </div>
      {lastError && <div style={styles.errorBanner}>{lastError}</div>}
    </div>
  )
}

function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span style={styles.statusWrap}>
      <span style={{ ...styles.dot, background: ok ? 'var(--good)' : 'var(--text-faint)' }} />
      <span style={styles.statusLabel}>{label}</span>
    </span>
  )
}

const styles: Record<string, React.CSSProperties> = {
  bar: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    gap: 24,
    height: 48,
    padding: '0 16px',
    background: 'var(--surface)',
    borderBottom: '1px solid var(--border)',
    flexShrink: 0,
  },
  left: { display: 'flex', alignItems: 'baseline', gap: 10 },
  wordmark: { fontWeight: 700, fontSize: 13, letterSpacing: 0.5, color: 'var(--text)' },
  subtitle: { fontSize: 11, color: 'var(--text-dim)' },
  center: { display: 'flex', alignItems: 'center', gap: 10, flex: 1 },
  meta: { fontSize: 11, color: 'var(--text-faint)' },
  right: { display: 'flex', alignItems: 'center', gap: 8 },
  select: {
    background: 'var(--surface-raised)',
    border: '1px solid var(--border-strong)',
    color: 'var(--text)',
    fontSize: 12,
    padding: '5px 8px',
    borderRadius: 3,
  },
  loadBtn: {
    background: 'var(--accent-dim)',
    border: '1px solid var(--accent)',
    color: 'var(--text)',
    fontSize: 12,
    padding: '6px 12px',
    borderRadius: 3,
  },
  statusWrap: { display: 'flex', alignItems: 'center', gap: 6 },
  dot: { width: 7, height: 7, borderRadius: '50%', display: 'inline-block' },
  statusLabel: { fontSize: 11, color: 'var(--text-dim)' },
  errorBanner: {
    position: 'absolute',
    top: 48,
    left: 0,
    right: 0,
    background: '#2a1416',
    color: 'var(--bad)',
    fontSize: 12,
    padding: '6px 16px',
    borderBottom: '1px solid var(--bad)',
    zIndex: 10,
  },
}
