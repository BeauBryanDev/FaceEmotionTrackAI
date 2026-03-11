
import { useState, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import { useBiometrics } from '../../context/Biometrics'

// -----------------------------------------------------------------------------
// PAGE TITLE MAP
// Maps route paths to cyberpunk display titles
// -----------------------------------------------------------------------------
const PAGE_TITLES = {
  '/dashboard': { title: 'LIVE STREAM',   sub: 'REAL-TIME BIOMETRIC FEED' },
  '/inference': { title: 'INFERENCE',     sub: 'REAL-TIME STREAMING' },
  '/history':   { title: 'HISTORY',       sub: 'EMOTION ARCHIVES' },
  '/emotions':  { title: 'EMOTION LOG',   sub: 'DETECTION HISTORY & ANALYTICS' },
  '/analytics':{ title: 'ANALYTICS',     sub: 'PCA EMBEDDING SPACE' },
  '/report':    { title: 'EMOTION REPORT', sub: 'OPERATIONAL EMOTION INTELLIGENCE' },
  '/russelquadrants': { title: 'EMOTION ANALYSIS', sub: 'RUSSELL CIRCUMPLEX MODEL' },
  '/pcaanalytics': { title: 'PCA ANALYTICS', sub: 'EMBEDDING SPACE VISUALIZATION' },
  '/profile':   { title: 'OPERATOR',      sub: 'IDENTITY & BIOMETRIC PROFILE' },
}

// -----------------------------------------------------------------------------
// SYSTEM CLOCK
// Live HH:MM:SS display updating every second
// -----------------------------------------------------------------------------
const SystemClock = () => {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const interval = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(interval)
  }, [])

  const pad = (n) => String(n).padStart(2, '0')
  const timeStr = `${pad(time.getHours())}:${pad(time.getMinutes())}:${pad(time.getSeconds())}`
  const dateStr = time.toISOString().slice(0, 10)

  return (
    <div style={{ textAlign: 'right' }}>
      <div style={{
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '1rem',
        color: '#cc44ff',
        textShadow: '0 0 8px rgba(170,0,255,0.5)',
        letterSpacing: '0.1em',
        lineHeight: 1,
      }}>
        {timeStr}
      </div>
      <div style={{
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.55rem',
        color: 'rgba(170,0,255,0.4)',
        letterSpacing: '0.15em',
        marginTop: '0.2rem',
      }}>
        {dateStr}
      </div>
    </div>
  )
}

// -----------------------------------------------------------------------------
// USER AVATAR
// Initials-based avatar with neon glow
// -----------------------------------------------------------------------------
const UserAvatar = ({ user, hasEmbedding }) => {
  const initials = user?.full_name
    ? user.full_name.split(' ').map((n) => n[0]).slice(0, 2).join('').toUpperCase()
    : 'OP'

  return (
    <div style={{
      width: 36,
      height: 36,
      background: 'linear-gradient(135deg, rgba(102,0,179,0.8), rgba(170,0,255,0.5))',
      border: '1px solid rgba(170,0,255,0.5)',
      boxShadow: '0 0 12px rgba(170,0,255,0.3)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'Orbitron, monospace',
      fontSize: '0.65rem',
      fontWeight: 700,
      color: '#f0ccff',
      letterSpacing: '0.05em',
      flexShrink: 0,
      position: 'relative',
    }}>
      {initials}
      {/* Biometric enrolled indicator */}
      {hasEmbedding && (
        <div style={{
          position: 'absolute',
          bottom: -2, right: -2,
          width: 8, height: 8,
          borderRadius: '50%',
          background: '#00ff88',
          boxShadow: '0 0 6px rgba(0,255,136,0.8)',
          border: '1px solid #0d0010',
        }} />
      )}
    </div>
  )
}

// -----------------------------------------------------------------------------
// HEADER
// -----------------------------------------------------------------------------
const Header = ({ onMenuClick }) => {
  const { user }   = useAuth()
  const { hasEmbedding } = useBiometrics()
  const location   = useLocation()
  const pageInfo   = PAGE_TITLES[location.pathname] || { title: 'SYSTEM', sub: 'FACETRACK_AI' }

  return (
    <header className="relative sticky top-0 z-50 w-full border-b border-purple-800/30 bg-[rgba(13,0,16,0.85)] backdrop-blur-md flex flex-col gap-3 px-4 py-3 sm:px-6 lg:px-8 md:flex-row md:items-center md:justify-between">

      {/* Left - page title */}
      <div className="flex items-center gap-3">
        {onMenuClick && (
          <button
            type="button"
            onClick={onMenuClick}
            className="md:hidden inline-flex h-8 w-8 items-center justify-center border border-purple-700/60 bg-purple-950/40 text-purple-200 hover:border-neon-purple/80"
            aria-label="Open menu"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
        )}
        {/* Vertical accent bar */}
        <div style={{
          width: '2px',
          height: '32px',
          background: 'linear-gradient(180deg, transparent, #bf00ff, transparent)',
          boxShadow: '0 0 8px rgba(170,0,255,0.6)',
          flexShrink: 0,
        }} />
        <div>
          <h1 style={{
            fontFamily: 'Orbitron, monospace',
            fontSize: '0.9rem',
            fontWeight: 900,
            color: '#f0ccff',
            letterSpacing: '0.2em',
            margin: 0,
            lineHeight: 1,
            textShadow: '0 0 12px rgba(170,0,255,0.3)',
          }}>
            {pageInfo.title}
          </h1>
          <div style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.55rem',
            color: 'rgba(170,0,255,0.45)',
            letterSpacing: '0.2em',
            marginTop: '0.25rem',
          }}>
            {pageInfo.sub}
          </div>
        </div>
      </div>

      {/* Right - clock + status indicators + avatar */}
      <div className="flex w-full flex-wrap items-center gap-4 md:w-auto md:flex-nowrap md:gap-6">

        {/* System metrics */}
        <div className="flex items-center gap-4">
          {[
            { label: 'MODELS', value: '4/4', ok: true },
            { label: 'STREAM', value: 'READY', ok: true },
          ].map(({ label, value, ok }) => (
            <div key={label} style={{ textAlign: 'center' }}>
              <div style={{
                fontFamily: 'Share Tech Mono, monospace',
                fontSize: '0.65rem',
                color: ok ? '#cc44ff' : 'rgba(255,80,80,0.8)',
                letterSpacing: '0.05em',
                textShadow: ok ? '0 0 6px rgba(170,0,255,0.4)' : 'none',
              }}>
                {value}
              </div>
              <div style={{
                fontFamily: 'Share Tech Mono, monospace',
                fontSize: '0.5rem',
                color: 'rgba(170,0,255,0.35)',
                letterSpacing: '0.15em',
              }}>
                {label}
              </div>
            </div>
          ))}
        </div>

        {/* Divider */}
        <div className="hidden sm:block w-px h-8 bg-purple-800/40" />

        {/* Clock */}
        <SystemClock />

        {/* Divider */}
        <div className="hidden sm:block w-px h-8 bg-purple-800/40" />

        {/* User avatar */}
        <UserAvatar user={user} hasEmbedding={hasEmbedding} />
      </div>

      {/* Bottom glow line */}
      <div style={{
        position: 'absolute',
        bottom: 0, left: 0, right: 0,
        height: '1px',
        background: 'linear-gradient(90deg, transparent, rgba(170,0,255,0.4), transparent)',
      }} />
    </header>
  )
}

export default Header
