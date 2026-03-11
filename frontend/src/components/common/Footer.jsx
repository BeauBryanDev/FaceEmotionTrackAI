
import { useState, useEffect } from 'react'

// -----------------------------------------------------------------------------
// TELEMETRY TICKER
// Scrolling status messages simulating a system log feed
// -----------------------------------------------------------------------------
const TICKER_MESSAGES = [
  'SCRFD DETECTION ENGINE NOMINAL',
  'ARCFACE RECOGNITION MODULE ACTIVE',
  'MINIFASNET LIVENESS GUARD ONLINE',
  'EFFICIENTNET EMOTION CLASSIFIER READY',
  'PGVECTOR DATABASE CONNECTION STABLE',
  'JWT AUTHENTICATION PROTOCOL ACTIVE',
  'WEBSOCKET STREAM CHANNEL OPEN',
  'EAR DROWSINESS MONITOR CALIBRATED',
  'HEAD POSE ESTIMATION CALIBRATED',
  'PCA ANALYTICS ENGINE STANDBY',
]

const Ticker = () => {
  const [index, setIndex] = useState(0)
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    const interval = setInterval(() => {
      setVisible(false)
      setTimeout(() => {
        setIndex((prev) => (prev + 1) % TICKER_MESSAGES.length)
        setVisible(true)
      }, 300)
    }, 3500)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex items-center gap-2 overflow-hidden max-w-full sm:max-w-[340px]">
      <span style={{
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.55rem',
        color: 'rgba(170,0,255,0.35)',
        letterSpacing: '0.1em',
        flexShrink: 0,
      }}>
        SYS /
      </span>
      <span style={{
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.58rem',
        color: 'rgba(170,0,255,0.55)',
        letterSpacing: '0.1em',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        opacity: visible ? 1 : 0,
        transition: 'opacity 0.3s ease',
      }}>
        {TICKER_MESSAGES[index]}
      </span>
    </div>
  )
}

// -----------------------------------------------------------------------------
// FOOTER
// -----------------------------------------------------------------------------
const Footer = () => {
  const [backendOk, setBackendOk] = useState(null)

  // Ping backend health endpoint on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
        const res    = await fetch(`${apiUrl}/health`, {
          signal: AbortSignal.timeout(3000),
        })
        setBackendOk(res.ok)
      } catch {
        setBackendOk(false)
      }
    }
    checkBackend()
  }, [])

  const backendColor = backendOk === null
    ? 'rgba(170,0,255,0.4)'
    : backendOk
      ? '#00ff88'
      : '#ff4466'

  const backendLabel = backendOk === null
    ? 'CONNECTING'
    : backendOk
      ? 'BACKEND ONLINE'
      : 'BACKEND OFFLINE'

  return (
    <footer className="relative flex flex-col gap-2 border-t border-purple-800/20 bg-[rgba(13,0,16,0.9)] px-4 py-2 sm:flex-row sm:items-center sm:justify-between sm:px-6">

      {/* Top glow line */}
      <div style={{
        position: 'absolute',
        top: 0, left: 0, right: 0,
        height: '1px',
        background: 'linear-gradient(90deg, transparent, rgba(170,0,255,0.25), transparent)',
      }} />

      {/* Left - telemetry ticker */}
      <Ticker />

      {/* Center - tech stack */}
      <div className="flex flex-wrap items-center gap-2 sm:justify-center md:absolute md:left-1/2 md:top-1/2 md:-translate-x-1/2 md:-translate-y-1/2">
        {['FastAPI', 'ONNX', 'pgvector', 'React'].map((tag) => (
          <span key={tag} style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.5rem',
            letterSpacing: '0.1em',
            padding: '2px 6px',
            border: '1px solid rgba(170,0,255,0.2)',
            color: 'rgba(170,0,255,0.4)',
            background: 'rgba(170,0,255,0.04)',
          }}>
            {tag}
          </span>
        ))}
      </div>

      {/* Right - backend status + version */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Backend status */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
        }}>
          <div style={{
            width: 5, height: 5,
            borderRadius: '50%',
            background: backendColor,
            boxShadow: backendOk ? `0 0 6px ${backendColor}` : 'none',
            animation: backendOk === null
              ? 'statusPulse 1s ease-in-out infinite'
              : 'none',
          }} />
          <span style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.55rem',
            letterSpacing: '0.1em',
            color: backendOk === null
              ? 'rgba(170,0,255,0.4)'
              : backendOk
                ? 'rgba(0,255,136,0.7)'
                : 'rgba(255,68,102,0.7)',
          }}>
            {backendLabel}
          </span>
        </div>

        {/* Version */}
        <span style={{
          fontFamily: 'Share Tech Mono, monospace',
          fontSize: '0.55rem',
          color: 'rgba(170,0,255,0.25)',
          letterSpacing: '0.1em',
        }}>
          v1.0.0
        </span>
      </div>

      <style>{`
        @keyframes statusPulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }
      `}</style>
    </footer>
  )
}

export default Footer
