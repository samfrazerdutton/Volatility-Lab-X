import { NavTabs } from './components/NavTabs'
import { TopBar } from './components/TopBar'
import { AppStateProvider, useAppState } from './state/AppState'
import { MarketExplorerView } from './views/MarketExplorerView'
import { OverviewView } from './views/OverviewView'
import { SurfaceLabView } from './views/SurfaceLabView'

function Shell() {
  const { activeView } = useAppState()
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <TopBar />
      <NavTabs />
      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        {activeView === 'overview' && <OverviewView />}
        {activeView === 'surface' && <SurfaceLabView />}
        {activeView === 'market' && <MarketExplorerView />}
      </div>
    </div>
  )
}

export default function App() {
  return (
    <AppStateProvider>
      <Shell />
    </AppStateProvider>
  )
}
