import type { ReactNode } from 'react'

export function Panel({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <div style={panelStyles.wrap}>
      <div style={panelStyles.header}>
        <span style={panelStyles.title}>{title}</span>
        {right}
      </div>
      <div style={panelStyles.body}>{children}</div>
    </div>
  )
}

const panelStyles: Record<string, React.CSSProperties> = {
  wrap: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 12px',
    borderBottom: '1px solid var(--border)',
    background: 'var(--surface-raised)',
  },
  title: {
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    color: 'var(--text-dim)',
  },
  body: { padding: 12 },
}

export function Metric({
  label,
  value,
  tone,
  definition,
}: {
  label: string
  value: ReactNode
  tone?: 'good' | 'bad' | 'warn'
  definition?: string
}) {
  return (
    <div style={metricStyles.wrap} title={definition}>
      <div style={metricStyles.label}>{label}</div>
      <div
        className="mono"
        style={{
          ...metricStyles.value,
          color: tone === 'good' ? 'var(--good)' : tone === 'bad' ? 'var(--bad)' : tone === 'warn' ? 'var(--warn)' : 'var(--text)',
        }}
      >
        {value}
      </div>
    </div>
  )
}

const metricStyles: Record<string, React.CSSProperties> = {
  wrap: { display: 'flex', flexDirection: 'column', gap: 3, minWidth: 110 },
  label: { fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: 0.4 },
  value: { fontSize: 16, fontWeight: 600 },
}

export function MetricGrid({ children }: { children: ReactNode }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20 }}>{children}</div>
}

export function StatusPill({ text, tone }: { text: string; tone: 'good' | 'bad' | 'warn' | 'neutral' }) {
  const color =
    tone === 'good' ? 'var(--good)' : tone === 'bad' ? 'var(--bad)' : tone === 'warn' ? 'var(--warn)' : 'var(--text-dim)'
  return (
    <span
      className="mono"
      style={{
        fontSize: 10,
        padding: '2px 7px',
        borderRadius: 3,
        border: `1px solid ${color}`,
        color,
        textTransform: 'uppercase',
      }}
    >
      {text}
    </span>
  )
}

export function EmptyState({ message }: { message: string }) {
  return <div style={{ color: 'var(--text-faint)', fontSize: 12, padding: 24, textAlign: 'center' }}>{message}</div>
}
