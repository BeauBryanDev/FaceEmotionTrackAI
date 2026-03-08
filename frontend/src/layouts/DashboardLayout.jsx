
import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

import Header from '../components/common/Header'
import Footer from '../components/common/Footer'

// -----------------------------------------------------------------------------
// NAV ITEMS
// -----------------------------------------------------------------------------
const NAV_ITEMS = [
  {
    path: '/about',
    label: 'ABOUT SYSTEM',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
    ),
  },
  {
    path: '/dashboard',
    label: 'LIVE STREAM',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" />
        <path d="M12 6v2M12 16v2M6 12H4M20 12h-2" />
      </svg>
    ),
  },
  {
    path: '/inference',
    label: 'INFERENCE',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M4 12h16" />
        <path d="M12 4v16" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    ),
  },
  {
    path: '/history',
    label: 'HISTORY',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v6l4 2" />
      </svg>
    ),
  },
  {
    path: '/emotions',
    label: 'EMOTION LOG',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M3 3h18v14H3z" />
        <path d="M7 17l5-5 3 3 4-6" />
      </svg>
    ),
  },
  {
    path: '/analytics',
    label: 'ANALYTICS',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M4 19V5" />
        <path d="M10 19V9" />
        <path d="M16 19v-6" />
        <path d="M22 19V3" />
      </svg>
    ),
  },
  {
    path: '/report',
    label: 'EMOTION REPORT',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M4 4h16v16H4z" />
        <path d="M8 16v-5" />
        <path d="M12 16V8" />
        <path d="M16 16v-2" />
      </svg>
    ),
  },
  {
    path: '/russelquadrants',
    label: 'EMOTION ANALYSIS',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M12 2v20" />
        <path d="M2 12h20" />
        <circle cx="16" cy="8" r="2.5" />
      </svg>
    ),
  },
  {
    path: '/pcaanalytics',
    label: 'PCA ANALYTICS',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M4 20V8" />
        <path d="M10 20V4" />
        <path d="M16 20v-6" />
        <path d="M22 20v-10" />
      </svg>
    ),
  },
  {
    path: '/profile',
    label: 'OPERATOR PROFILE',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="8" r="4" />
        <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
      </svg>
    ),
  },
]

// -----------------------------------------------------------------------------
// SIDEBAR
// -----------------------------------------------------------------------------
const Sidebar = () => {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <aside style={{
      position: 'fixed',
      top: 0, left: 0, bottom: 0,
      width: '240px',
      background: 'linear-gradient(180deg, #130020 0%, #0d0010 100%)',
      borderRight: '1px solid rgba(170,0,255,0.2)',
      display: 'flex',
      flexDirection: 'column',
      zIndex: 100,
      overflowY: 'auto',
    }}>

      {/* Top edge glow */}
      <div style={{
        position: 'absolute',
        top: 0, left: 0, right: 0,
        height: '1px',
        background: 'linear-gradient(90deg, transparent, rgba(170,0,255,0.8), transparent)',
      }} />

      {/* Brand */}
      <div style={{
        padding: '1.5rem 1.25rem 1.25rem',
        borderBottom: '1px solid rgba(170,0,255,0.1)',
      }}>
        <div style={{
          fontFamily: 'Orbitron, monospace',
          fontSize: '0.75rem',
          fontWeight: 900,
          color: '#bf00ff',
          textShadow: '0 0 16px rgba(170,0,255,0.6)',
          letterSpacing: '0.2em',
          lineHeight: 1.3,
        }}>
          FACE<br />TRACK_AI
        </div>
        <div style={{
          fontFamily: 'Share Tech Mono, monospace',
          fontSize: '0.55rem',
          color: 'rgba(170,0,255,0.4)',
          letterSpacing: '0.15em',
          marginTop: '0.3rem',
        }}>
          BIOMETRIC SYSTEM v1.0
        </div>
      </div>

      {/* System status */}
      <div style={{
        margin: '1rem 1.25rem',
        padding: '0.6rem 0.75rem',
        background: 'rgba(170,0,255,0.05)',
        border: '1px solid rgba(170,0,255,0.12)',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
      }}>
        <div style={{
          width: 6, height: 6, borderRadius: '50%',
          background: '#bf00ff',
          boxShadow: '0 0 6px rgba(170,0,255,0.8)',
          flexShrink: 0,
          animation: 'statusPulse 2s ease-in-out infinite',
        }} />
        <span style={{
          fontFamily: 'Share Tech Mono, monospace',
          fontSize: '0.6rem',
          letterSpacing: '0.1em',
          color: 'rgba(170,0,255,0.6)',
        }}>
          NEURAL CORE ONLINE
        </span>
      </div>

      {/* Nav section label */}
      <div style={{
        padding: '0 1.25rem 0.5rem',
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.55rem',
        letterSpacing: '0.2em',
        color: 'rgba(170,0,255,0.3)',
        textTransform: 'uppercase',
      }}>
        MODULES
      </div>

      {/* Nav links */}
      <nav style={{ flex: 1, padding: '0 0.75rem' }}>
        {NAV_ITEMS.map(({ path, label, icon }) => (
          <NavLink
            key={path}
            to={path}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              padding: '0.8rem 0.8rem',
              marginBottom: '0.25rem',
              textDecoration: 'none',
              fontFamily: 'Orbitron, monospace',
              fontSize: '0.78rem',
              fontWeight: 700,
              letterSpacing: '0.12em',
              color: isActive ? '#f0ccff' : '#d9a2ff',
              background: isActive ? 'rgba(170,0,255,0.16)' : 'transparent',
              border: isActive
                ? '1px solid rgba(220,150,255,0.55)'
                : '1px solid transparent',
              borderLeft: isActive
                ? '3px solid #cc88ff'
                : '2px solid transparent',
              transition: 'all 0.2s',
              textShadow: isActive
                ? '0 0 12px rgba(220,150,255,0.7)'
                : '0 0 8px rgba(190,120,255,0.45)',
              boxShadow: isActive ? '0 0 16px rgba(190,120,255,0.22)' : 'none',
            })}
          >
            <span style={{ opacity: 0.95, flexShrink: 0 }}>{icon}</span>
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Bottom - user info + logout */}
      <div style={{
        borderTop: '1px solid rgba(170,0,255,0.1)',
        padding: '1rem 1.25rem',
      }}>

        {/* User info */}
        <div style={{
          marginBottom: '0.75rem',
          padding: '0.6rem 0.75rem',
          background: 'rgba(45,0,87,0.3)',
          border: '1px solid rgba(170,0,255,0.15)',
        }}>
          <div style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.6rem',
            letterSpacing: '0.1em',
            color: 'rgba(170,0,255,0.4)',
            marginBottom: '0.2rem',
          }}>
            OPERATOR
          </div>
          <div style={{
            fontFamily: 'Orbitron, monospace',
            fontSize: '0.65rem',
            fontWeight: 600,
            color: '#cc44ff',
            letterSpacing: '0.05em',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            {user?.full_name || 'UNKNOWN'}
          </div>
          <div style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.55rem',
            color: 'rgba(170,0,255,0.35)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            marginTop: '0.1rem',
          }}>
            {user?.email || ''}
          </div>
        </div>

        {/* Logout */}
        <button
          onClick={handleLogout}
          style={{
            width: '100%',
            padding: '0.6rem',
            background: 'transparent',
            border: '1px solid rgba(255,0,80,0.25)',
            color: 'rgba(255,80,120,0.6)',
            fontFamily: 'Orbitron, monospace',
            fontSize: '0.6rem',
            fontWeight: 700,
            letterSpacing: '0.2em',
            textTransform: 'uppercase',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => {
            e.target.style.background = 'rgba(255,0,80,0.1)'
            e.target.style.borderColor = 'rgba(255,0,80,0.5)'
            e.target.style.color = 'rgba(255,80,120,0.9)'
            e.target.style.boxShadow = '0 0 12px rgba(255,0,80,0.15)'
          }}
          onMouseLeave={(e) => {
            e.target.style.background = 'transparent'
            e.target.style.borderColor = 'rgba(255,0,80,0.25)'
            e.target.style.color = 'rgba(255,80,120,0.6)'
            e.target.style.boxShadow = 'none'
          }}
        >
          DISCONNECT
        </button>
      </div>

      <style>{`
        @keyframes statusPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.4; transform: scale(0.85); }
        }
      `}</style>
    </aside>
  )
}

// -----------------------------------------------------------------------------
// DASHBOARD LAYOUT

// -----------------------------------------------------------------------------
const DashboardLayout = () => (
  <div style={{
    display: 'flex',
    minHeight: '100vh',
    background: 'var(--purple-950)',
  }}>
    <Sidebar />

    {/* Right column */}
    <div style={{
      marginLeft: '240px',
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      minHeight: '100vh',
    }}>

      {/* Background grid fixed behind all content */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: '240px',
        right: 0,
        bottom: 0,
        backgroundImage: `
          linear-gradient(rgba(170,0,255,0.025) 1px, transparent 1px),
          linear-gradient(90deg, rgba(170,0,255,0.025) 1px, transparent 1px)
        `,
        backgroundSize: '40px 40px',
        pointerEvents: 'none',
        zIndex: 0,
      }} />

      {/* Header sticky at top of right column */}
      <Header />

      {/* Page content fills remaining vertical space */}
      <main style={{
        flex: 1,
        padding: '2rem',
        position: 'relative',
        zIndex: 1,
      }}>
        <Outlet />
      </main>

      {/* Footer always at bottom of right column */}
      <Footer />
    </div>
  </div>
)

export default DashboardLayout
