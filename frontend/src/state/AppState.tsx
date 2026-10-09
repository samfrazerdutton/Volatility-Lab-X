import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type {
  ApplyEventResponse,
  CompareResponse,
  EventLogEntry,
  ExpiryInfo,
  HealthResponse,
  Quote,
} from '../api/types'

export type ViewId =
  | 'overview'
  | 'market'
  | 'surface'
  | 'graph'
  | 'performance'
  | 'validation'
  | 'replay'

interface AppStateShape {
  health: HealthResponse | null
  expiries: ExpiryInfo[]
  selectedExpiry: number | null
  selectedQuote: Quote | null
  lastApply: ApplyEventResponse | null
  lastCompare: CompareResponse | null
  events: EventLogEntry[]
  activeView: ViewId
  loading: boolean
  lastError: string | null

  setActiveView: (v: ViewId) => void
  setSelectedExpiry: (y: number | null) => void
  setSelectedQuote: (q: Quote | null) => void
  refreshHealth: () => Promise<void>
  refreshExpiries: () => Promise<void>
  refreshEvents: () => Promise<void>
  reloadMarket: (regime: string) => Promise<void>
  recordApply: (r: ApplyEventResponse) => void
  recordCompare: (r: CompareResponse) => void
  setError: (msg: string | null) => void
}

const Ctx = createContext<AppStateShape | null>(null)

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [expiries, setExpiries] = useState<ExpiryInfo[]>([])
  const [selectedExpiry, setSelectedExpiry] = useState<number | null>(null)
  const [selectedQuote, setSelectedQuote] = useState<Quote | null>(null)
  const [lastApply, setLastApply] = useState<ApplyEventResponse | null>(null)
  const [lastCompare, setLastCompare] = useState<CompareResponse | null>(null)
  const [events, setEvents] = useState<EventLogEntry[]>([])
  const [activeView, setActiveView] = useState<ViewId>('overview')
  const [loading, setLoading] = useState(false)
  const [lastError, setLastError] = useState<string | null>(null)

  const refreshHealth = useCallback(async () => {
    setHealth(await api.health())
  }, [])

  const refreshExpiries = useCallback(async () => {
    const res = await api.expiries()
    setExpiries(res.expiries)
  }, [])

  const refreshEvents = useCallback(async () => {
    const res = await api.events()
    setEvents(res.events)
  }, [])

  const reloadMarket = useCallback(
    async (regime: string) => {
      setLoading(true)
      setLastError(null)
      try {
        await api.loadMarket(regime)
        setSelectedQuote(null)
        setLastApply(null)
        setLastCompare(null)
        await Promise.all([refreshHealth(), refreshExpiries(), refreshEvents()])
      } catch (e) {
        setLastError(e instanceof Error ? e.message : String(e))
      } finally {
        setLoading(false)
      }
    },
    [refreshHealth, refreshExpiries, refreshEvents],
  )

  const value: AppStateShape = {
    health,
    expiries,
    selectedExpiry,
    selectedQuote,
    lastApply,
    lastCompare,
    events,
    activeView,
    loading,
    lastError,
    setActiveView,
    setSelectedExpiry,
    setSelectedQuote,
    refreshHealth,
    refreshExpiries,
    refreshEvents,
    reloadMarket,
    recordApply: setLastApply,
    recordCompare: setLastCompare,
    setError: setLastError,
  }

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useAppState(): AppStateShape {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useAppState must be used within AppStateProvider')
  return ctx
}
