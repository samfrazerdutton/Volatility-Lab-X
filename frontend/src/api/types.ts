// Mirrors the JSON contract in apps/server/main.cpp exactly -- every field
// here corresponds to a real field the C++ service serializes from a real
// engine type (OptionQuote, SviFitResult, RecomputeReport, PortfolioGreeks,
// PnLAttribution, OptionGreeks). Nothing in this file is invented: if a
// field is added here, it must first exist on the server response.

export interface HealthResponse {
  status: string
  compiler: string
  compiler_version: string
  build_type: string
  simd: string
  host_cpu: string
  market_loaded: boolean
}

export interface LoadMarketResponse {
  loaded: boolean
  regime: string
  spot: number
  quotes_total: number
  expiries: number[]
  state_hash: string
}

export interface Quote {
  years: number
  strike: number
  type: 'call' | 'put'
  bid: number
  ask: number
  mid: number
  implied_vol: number
  volume: number
  open_interest: number
  age_seconds: number
  status: 'unvalidated' | 'ok' | 'degraded' | 'rejected'
  weight: number
  log_moneyness: number
  forward: number
}

export interface QuotesResponse {
  quotes: Quote[]
  matched: number
  returned: number
}

export interface ExpiryInfo {
  years: number
  quote_count: number
}

export interface ExpiriesResponse {
  expiries: ExpiryInfo[]
}

export interface SurfacePoint {
  strike: number
  log_moneyness: number
  implied_vol: number
  bid: number
  ask: number
  mid: number
  status: string
  type: string
}

export interface CalibratedPoint {
  log_moneyness: number
  implied_vol: number
}

export interface FitDiagnostics {
  status: string
  ok: boolean
  objective: number
  rms_vol_error: number
  max_vol_error: number
  quotes_used: number
  quotes_available: number
  inner_solves: number
  outer_iterations: number
  active_constraints: number
}

export interface SurfaceResponse {
  years: number
  observed: SurfacePoint[]
  calibrated: CalibratedPoint[]
  fit: FitDiagnostics
}

export interface Greeks {
  price: number
  delta: number
  gamma: number
  vega: number
  theta: number
  rho: number
  vanna: number
  volga: number
  charm: number
  speed: number
}

export interface RecomputeReportDto {
  total_nodes: number
  recomputed_nodes: number
  reused_nodes: number
  fraction_avoided: number
  quotes_total: number
  quotes_examined: number
  calibrations_run: number
  latency_us: number
}

export interface PortfolioDto {
  value: number
  delta: number
  gamma: number
  vega: number
  theta: number
  rho: number
}

export interface PnlDto {
  total_exact_pnl: number
  base_value: number
  new_value: number
  spot_pnl: number
  vol_pnl: number
  rate_pnl: number
  theta_pnl: number
  gamma_pnl: number
  residual: number
}

export interface PositionResult {
  label: string
  strike: number
  years: number
  type: string
  vol_used: number
  greeks: Greeks
}

export interface ApplyEventResponse {
  applied: boolean
  report: RecomputeReportDto
  portfolio: PortfolioDto
  pnl: PnlDto
  positions: PositionResult[]
  state_hash: string
}

export interface ApiError {
  error: string
  message?: string
  field?: string
  param?: string
}

export interface SurfaceDiff {
  years: number
  incremental_vol: number
  full_rebuild_vol: number
  diff: number
}

export interface CompareResponse {
  incremental_us: number
  incremental_measured: boolean
  full_rebuild_ms: number
  speedup: number | null
  pnl_diff: number
  max_vol_diff: number
  surface_diffs: SurfaceDiff[]
  quotes_examined: number
  quotes_total: number
  nodes_total: number
  nodes_recomputed: number
}

export interface EventLogEntry {
  timestamp_ms: number
  type: string
  description: string
  state_hash: string
}

export interface EventsResponse {
  events: EventLogEntry[]
}

export type GreeksResponse = Greeks
