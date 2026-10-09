import type { CalibratedPoint, SurfacePoint } from '../api/types'

// A deliberately plain, hand-rolled SVG chart rather than a charting
// library: this view needs exactly one scatter-plus-line plot, and pulling
// in a charting dependency for one plot is the "unnecessary dependency"
// the directive asks to avoid. The 3D WebGL surface (Surface Lab's
// flagship view, directive section 6) is a separate, larger piece of work
// and is explicitly not implemented in this build -- see README.

interface Props {
  observed: SurfacePoint[]
  calibrated: CalibratedPoint[]
  onSelect?: (p: SurfacePoint) => void
  selected?: SurfacePoint | null
}

const W = 640
const H = 320
const PAD = { top: 16, right: 16, bottom: 32, left: 48 }

export function SmileChart({ observed, calibrated, onSelect, selected }: Props) {
  if (observed.length === 0 && calibrated.length === 0) {
    return <div style={{ color: 'var(--text-faint)', fontSize: 12 }}>No data for this expiry.</div>
  }

  const allK = [...observed.map((o) => o.log_moneyness), ...calibrated.map((c) => c.log_moneyness)]
  const allVol = [...observed.map((o) => o.implied_vol), ...calibrated.map((c) => c.implied_vol)]
  const kMin = Math.min(...allK)
  const kMax = Math.max(...allK)
  const volMin = Math.min(...allVol) * 0.95
  const volMax = Math.max(...allVol) * 1.05

  const x = (k: number) =>
    PAD.left + ((k - kMin) / (kMax - kMin || 1)) * (W - PAD.left - PAD.right)
  const y = (v: number) =>
    H - PAD.bottom - ((v - volMin) / (volMax - volMin || 1)) * (H - PAD.top - PAD.bottom)

  const linePath = calibrated
    .slice()
    .sort((a, b) => a.log_moneyness - b.log_moneyness)
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(p.log_moneyness)} ${y(p.implied_vol)}`)
    .join(' ')

  const statusColor = (s: string) =>
    s === 'ok' ? 'var(--accent)' : s === 'degraded' ? 'var(--warn)' : s === 'rejected' ? 'var(--bad)' : 'var(--text-faint)'

  const yTicks = 5
  const xTicks = 5

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Volatility smile for the selected expiry">
      {/* grid */}
      {Array.from({ length: yTicks + 1 }, (_, i) => {
        const v = volMin + ((volMax - volMin) * i) / yTicks
        return (
          <g key={`y${i}`}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)} stroke="var(--border)" strokeWidth={1} />
            <text x={PAD.left - 8} y={y(v)} fontSize={10} fill="var(--text-faint)" textAnchor="end" dominantBaseline="middle">
              {(v * 100).toFixed(1)}%
            </text>
          </g>
        )
      })}
      {Array.from({ length: xTicks + 1 }, (_, i) => {
        const k = kMin + ((kMax - kMin) * i) / xTicks
        return (
          <g key={`x${i}`}>
            <line x1={x(k)} x2={x(k)} y1={PAD.top} y2={H - PAD.bottom} stroke="var(--border)" strokeWidth={1} />
            <text x={x(k)} y={H - PAD.bottom + 16} fontSize={10} fill="var(--text-faint)" textAnchor="middle">
              {k.toFixed(2)}
            </text>
          </g>
        )
      })}
      <text x={W / 2} y={H - 4} fontSize={10} fill="var(--text-dim)" textAnchor="middle">
        log-moneyness, k = log(K/F)
      </text>
      <text x={14} y={H / 2} fontSize={10} fill="var(--text-dim)" textAnchor="middle" transform={`rotate(-90 14 ${H / 2})`}>
        implied vol
      </text>

      {/* calibrated model curve -- explicitly a model estimate */}
      {linePath && <path d={linePath} fill="none" stroke="var(--accent-dim)" strokeWidth={2} />}

      {/* observed quotes -- explicitly market observations */}
      {observed.map((p, i) => {
        const isSelected = selected && selected.strike === p.strike && selected.type === p.type
        return (
          <circle
            key={i}
            cx={x(p.log_moneyness)}
            cy={y(p.implied_vol)}
            r={isSelected ? 6 : 4}
            fill={statusColor(p.status)}
            stroke={isSelected ? 'var(--text)' : 'none'}
            strokeWidth={2}
            style={{ cursor: onSelect ? 'pointer' : 'default' }}
            onClick={() => onSelect?.(p)}
          >
            <title>
              {`${p.type} K=${p.strike} vol=${(p.implied_vol * 100).toFixed(2)}% bid=${p.bid} ask=${p.ask} status=${p.status}`}
            </title>
          </circle>
        )
      })}

      {/* legend */}
      <g transform={`translate(${W - 170}, ${PAD.top})`}>
        <circle cx={0} cy={0} r={4} fill="var(--accent)" />
        <text x={10} y={4} fontSize={10} fill="var(--text-dim)">
          observed quote (ok)
        </text>
        <line x1={0} x2={14} y1={16} y2={16} stroke="var(--accent-dim)" strokeWidth={2} />
        <text x={20} y={20} fontSize={10} fill="var(--text-dim)">
          calibrated SVI (model)
        </text>
      </g>
    </svg>
  )
}
