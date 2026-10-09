import { useAppState, type ViewId } from '../state/AppState'

// Every tab the directive's shell asks for is listed -- the ones not yet
// wired to a real view are visibly disabled with a reason, rather than
// hidden (so the UI never implies a capability that was not built) or
// silently functional-looking but empty (which would be the "decorative
// control that does nothing" the directive explicitly forbids).
const TABS: { id: ViewId; label: string; implemented: boolean }[] = [
  { id: 'overview', label: 'Overview', implemented: true },
  { id: 'surface', label: 'Surface Lab', implemented: true },
  { id: 'market', label: 'Market Explorer', implemented: true },
  { id: 'graph', label: 'Compute Graph', implemented: false },
  { id: 'performance', label: 'Performance Lab', implemented: false },
  { id: 'validation', label: 'Validation', implemented: false },
  { id: 'replay', label: 'Replay Studio', implemented: false },
]

export function NavTabs() {
  const { activeView, setActiveView } = useAppState()
  return (
    <div style={styles.nav}>
      {TABS.map((t) => (
        <button
          key={t.id}
          disabled={!t.implemented}
          title={t.implemented ? undefined : 'Not implemented in this build -- see README'}
          onClick={() => t.implemented && setActiveView(t.id)}
          style={{
            ...styles.tab,
            ...(activeView === t.id ? styles.tabActive : {}),
            ...(t.implemented ? {} : styles.tabDisabled),
          }}
        >
          {t.label}
          {!t.implemented && <span style={styles.soon}>not built</span>}
        </button>
      ))}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  nav: {
    display: 'flex',
    gap: 2,
    padding: '0 12px',
    background: 'var(--surface)',
    borderBottom: '1px solid var(--border)',
    flexShrink: 0,
  },
  tab: {
    background: 'transparent',
    border: 'none',
    borderBottom: '2px solid transparent',
    color: 'var(--text-dim)',
    fontSize: 12,
    padding: '10px 14px',
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  tabActive: {
    color: 'var(--text)',
    borderBottom: '2px solid var(--accent)',
  },
  tabDisabled: {
    color: 'var(--text-faint)',
    cursor: 'not-allowed',
  },
  soon: {
    fontSize: 9,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    background: 'var(--surface-raised)',
    border: '1px solid var(--border-strong)',
    borderRadius: 2,
    padding: '1px 4px',
  },
}
