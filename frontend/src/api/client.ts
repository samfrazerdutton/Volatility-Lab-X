import type {
  ApiError,
  ApplyEventResponse,
  CompareResponse,
  EventsResponse,
  ExpiriesResponse,
  GreeksResponse,
  HealthResponse,
  LoadMarketResponse,
  Quote,
  QuotesResponse,
  SurfaceResponse,
} from './types'

export class ApiRequestError extends Error {
  status: number
  body: ApiError

  constructor(status: number, body: ApiError) {
    super(body.message ?? body.error)
    this.status = status
    this.body = body
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    let body: ApiError
    try {
      body = (await res.json()) as ApiError
    } catch {
      body = { error: 'unknown_error', message: `HTTP ${res.status}` }
    }
    throw new ApiRequestError(res.status, body)
  }
  return (await res.json()) as T
}

export const api = {
  health: () => request<HealthResponse>('/health'),

  loadMarket: (regime: string) =>
    request<LoadMarketResponse>('/market/load', {
      method: 'POST',
      body: JSON.stringify({ regime }),
    }),

  quotes: (opts?: { expiry?: number; type?: string; limit?: number; offset?: number }) => {
    const params = new URLSearchParams()
    if (opts?.expiry !== undefined) params.set('expiry', String(opts.expiry))
    if (opts?.type) params.set('type', opts.type)
    if (opts?.limit !== undefined) params.set('limit', String(opts.limit))
    if (opts?.offset !== undefined) params.set('offset', String(opts.offset))
    const qs = params.toString()
    return request<QuotesResponse>(`/market/quotes${qs ? `?${qs}` : ''}`)
  },

  expiries: () => request<ExpiriesResponse>('/market/expiries'),

  surface: (years: number) => request<SurfaceResponse>(`/surface?years=${years}`),

  applyEvent: (evt: {
    years: number
    strike: number
    option_type: 'call' | 'put'
    bid: number
    ask: number
    mid?: number
  }) =>
    request<ApplyEventResponse>('/event/apply', {
      method: 'POST',
      body: JSON.stringify(evt),
    }),

  compareFullRebuild: () =>
    request<CompareResponse>('/compare/full-rebuild', { method: 'POST' }),

  events: () => request<EventsResponse>('/events'),

  quoteGreeks: (q: { years: number; strike: number; type: string; vol: number }) =>
    request<GreeksResponse>(
      `/quote/greeks?years=${q.years}&strike=${q.strike}&type=${q.type}&vol=${q.vol}`,
    ),
}

export type { Quote }
