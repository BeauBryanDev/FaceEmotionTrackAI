import { Outlet } from 'react-router-dom'
import { useEffect, useRef } from 'react'

// -----------------------------------------------------------------------------
// ANIMATED GRID CANVAS
// Draws a perspective cyber-grid that pulses with purple glow
// -----------------------------------------------------------------------------
const CyberGridCanvas = () => {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let animFrame
    let tick = 0

    const resize = () => {
      canvas.width  = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const draw = () => {
      const { width, height } = canvas
      ctx.clearRect(0, 0, width, height)

      tick += 0.008
      const pulse = 0.4 + Math.sin(tick) * 0.15

      // Horizontal lines with perspective
      const lines = 18
      for (let i = 0; i <= lines; i++) {
        const y = (i / lines) * height
        const alpha = pulse * (0.15 + (i / lines) * 0.45)
        ctx.strokeStyle = `rgba(170, 0, 255, ${alpha})`
        ctx.lineWidth = i === 0 || i === lines ? 0.5 : 0.3
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(width, y)
        ctx.stroke()
      }

      // Vertical lines converging to vanishing point
      const vx = width * 0.5
      const vy = height * 0.42
      const vlines = 14
      for (let i = 0; i <= vlines; i++) {
        const x = (i / vlines) * width
        const alpha = pulse * 0.25
        ctx.strokeStyle = `rgba(170, 0, 255, ${alpha})`
        ctx.lineWidth = 0.3
        ctx.beginPath()
        ctx.moveTo(vx, vy)
        ctx.lineTo(x, height)
        ctx.stroke()
      }

      // Scanning horizontal beam
      const scanY = ((tick * 60) % height)
      const scanGrad = ctx.createLinearGradient(0, scanY - 12, 0, scanY + 12)
      scanGrad.addColorStop(0,   'rgba(170, 0, 255, 0)')
      scanGrad.addColorStop(0.5, `rgba(170, 0, 255, ${pulse * 0.35})`)
      scanGrad.addColorStop(1,   'rgba(170, 0, 255, 0)')
      ctx.fillStyle = scanGrad
      ctx.fillRect(0, scanY - 12, width, 24)

      animFrame = requestAnimationFrame(draw)
    }

    draw()
    return () => {
      cancelAnimationFrame(animFrame)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        opacity: 0.85,
      }}
    />
  )
}

// -----------------------------------------------------------------------------
// BRAND PANEL (left side)
// -----------------------------------------------------------------------------
const BrandPanel = () => (
  <div style={{
    position: 'relative',
    width: '50%',
    minHeight: '100vh',
    background: 'linear-gradient(160deg, #0d0010 0%, #1a0030 50%, #0d0010 100%)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    borderRight: '1px solid rgba(170, 0, 255, 0.2)',
  }}>
    <CyberGridCanvas />

    {/* Corner decorations */}
    <div style={{ position: 'absolute', top: 20, left: 20, width: 40, height: 40,
      borderTop: '1px solid rgba(170,0,255,0.6)',
      borderLeft: '1px solid rgba(170,0,255,0.6)' }} />
    <div style={{ position: 'absolute', top: 20, right: 20, width: 40, height: 40,
      borderTop: '1px solid rgba(170,0,255,0.6)',
      borderRight: '1px solid rgba(170,0,255,0.6)' }} />
    <div style={{ position: 'absolute', bottom: 20, left: 20, width: 40, height: 40,
      borderBottom: '1px solid rgba(170,0,255,0.6)',
      borderLeft: '1px solid rgba(170,0,255,0.6)' }} />
    <div style={{ position: 'absolute', bottom: 20, right: 20, width: 40, height: 40,
      borderBottom: '1px solid rgba(170,0,255,0.6)',
      borderRight: '1px solid rgba(170,0,255,0.6)' }} />

    {/* Brand content */}
    <div style={{ position: 'relative', zIndex: 2, textAlign: 'center', padding: '2rem' }}>

      {/* Hexagon icon */}
      <div style={{
        width: 96,
        height: 96,
        margin: '0 auto 2rem',
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <svg viewBox="0 0 100 100" style={{ width: '100%', height: '100%', filter: 'drop-shadow(0 0 12px rgba(170,0,255,0.8))' }}>
          <polygon
            points="50,4 93,27 93,73 50,96 7,73 7,27"
            fill="none"
            stroke="rgba(170,0,255,0.9)"
            strokeWidth="1.5"
          />
          <polygon
            points="50,18 80,34 80,66 50,82 20,66 20,34"
            fill="rgba(170,0,255,0.08)"
            stroke="rgba(170,0,255,0.4)"
            strokeWidth="1"
          />
          {/* Eye / face scan icon inside */}
          <ellipse cx="50" cy="50" rx="14" ry="9"
            fill="none" stroke="rgba(204,68,255,0.9)" strokeWidth="1.5" />
          <circle cx="50" cy="50" r="4"
            fill="rgba(191,0,255,0.9)" />
          <circle cx="50" cy="50" r="1.5"
            fill="white" />
          {/* Scan lines */}
          <line x1="30" y1="50" x2="36" y2="50" stroke="rgba(170,0,255,0.5)" strokeWidth="1" />
          <line x1="64" y1="50" x2="70" y2="50" stroke="rgba(170,0,255,0.5)" strokeWidth="1" />
        </svg>
      </div>

      {/* Title */}
      <h1 style={{
        fontFamily: 'Orbitron, monospace',
        fontSize: '1.6rem',
        fontWeight: 900,
        color: '#bf00ff',
        textShadow: '0 0 20px rgba(170,0,255,0.5), 0 0 40px rgba(170,0,255,0.2)',
        letterSpacing: '0.15em',
        marginBottom: '0.5rem',
        lineHeight: 1.1,
      }}>
        FACE<br />EMOTION<br />TRACK_AI
      </h1>

      {/* Subtitle */}
      <p style={{
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.7rem',
        color: 'rgba(170,0,255,0.7)',
        letterSpacing: '0.2em',
        textTransform: 'uppercase',
        marginBottom: '3rem',
      }}>
        BIOMETRIC ANALYSIS SYSTEM v1.0
      </p>

      {/* Stats row */}
      <div style={{
        display: 'flex',
        gap: '2rem',
        justifyContent: 'center',
        marginBottom: '2rem',
        flexWrap: 'wrap',
      }}>
        {[
          { label: 'MODELS', value: '4' },
          { label: 'DIMS',   value: '512' },
          { label: 'FPS',    value: '30' },
        ].map(({ label, value }) => (
          <div key={label} style={{ textAlign: 'center' }}>
            <div style={{
              fontFamily: 'Orbitron, monospace',
              fontSize: '1.4rem',
              fontWeight: 700,
              color: '#cc44ff',
              textShadow: '0 0 8px rgba(170,0,255,0.6)',
            }}>
              {value}
            </div>
            <div style={{
              fontFamily: 'Share Tech Mono, monospace',
              fontSize: '0.6rem',
              color: 'rgba(170,0,255,0.5)',
              letterSpacing: '0.15em',
            }}>
              {label}
            </div>
          </div>
        ))}
      </div>

      {/* Tech stack tags */}
      <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', flexWrap: 'wrap' }}>
        {['SCRFD', 'ArcFace', 'MiniFASNet', 'EfficientNet'].map((tag) => (
          <span key={tag} style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.6rem',
            letterSpacing: '0.1em',
            padding: '3px 8px',
            border: '1px solid rgba(170,0,255,0.3)',
            color: 'rgba(170,0,255,0.6)',
            background: 'rgba(170,0,255,0.05)',
          }}>
            {tag}
          </span>
        ))}
      </div>
    </div>

    {/* Bottom status bar */}
    <div style={{
      position: 'absolute',
      bottom: 0, left: 0, right: 0,
      padding: '0.75rem 1.5rem',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      borderTop: '1px solid rgba(170,0,255,0.15)',
      background: 'rgba(13,0,16,0.8)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <div style={{
          width: 6, height: 6, borderRadius: '50%',
          background: '#bf00ff',
          boxShadow: '0 0 6px rgba(170,0,255,0.8)',
          animation: 'statusPulse 2s ease-in-out infinite',
        }} />
        <span style={{
          fontFamily: 'Share Tech Mono, monospace',
          fontSize: '0.6rem',
          color: 'rgba(170,0,255,0.6)',
          letterSpacing: '0.1em',
        }}>
          SYSTEM ONLINE
        </span>
      </div>
      <span style={{
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.6rem',
        color: 'rgba(170,0,255,0.3)',
        letterSpacing: '0.1em',
      }}>
        NODE_01 / ACTIVE
      </span>
    </div>
  </div>
)

// -----------------------------------------------------------------------------
// AUTH LAYOUT
// -----------------------------------------------------------------------------
const AuthLayout = () => (
  <div className="flex min-h-screen flex-col bg-[var(--purple-950)] md:flex-row">
    {/* Left - Brand panel (hidden on mobile) */}
    <div style={{ display: 'flex', flex: '0 0 50%' }}
      className="hidden md:flex">
      <BrandPanel />
    </div>

    {/* Right - Form panel */}
    <div className="relative flex flex-1 flex-col items-center justify-center overflow-y-auto bg-gradient-to-b from-[#0d0010] to-[#130020] px-4 py-8 sm:px-6 md:px-10">
      {/* Subtle background grid */}
      <div style={{
        position: 'absolute',
        inset: 0,
        backgroundImage: `
          linear-gradient(rgba(170,0,255,0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(170,0,255,0.03) 1px, transparent 1px)
        `,
        backgroundSize: '40px 40px',
        pointerEvents: 'none',
      }} />

      {/* Mobile brand header */}
      <div className="flex md:hidden" style={{
        position: 'absolute',
        top: '1.25rem',
        left: '50%',
        transform: 'translateX(-50%)',
        fontFamily: 'Orbitron, monospace',
        fontSize: '0.9rem',
        fontWeight: 900,
        color: '#bf00ff',
        textShadow: '0 0 12px rgba(170,0,255,0.5)',
        letterSpacing: '0.2em',
        whiteSpace: 'nowrap',
      }}>
        FACETRACK_AI
      </div>

      {/* Form content */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        width: '100%',
        maxWidth: '420px',
        paddingTop: '3.5rem',
      }}>
        <Outlet />
      </div>
    </div>
  </div>
)

export default AuthLayout
