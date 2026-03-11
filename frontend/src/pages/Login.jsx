import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const Login = () => {
  const navigate        = useNavigate()
  const { login }       = useAuth()

  const [form, setForm]       = useState({ email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [phase, setPhase]     = useState('IDLE') // IDLE | SCANNING | AUTHENTICATING | SUCCESS

  const handleChange = (e) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }))
    if (error) setError(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.email || !form.password) {
      setError('ALL FIELDS REQUIRED')
      return
    }

    setLoading(true)
    setError(null)
    setPhase('SCANNING')

    try {
      setTimeout(() => setPhase('AUTHENTICATING'), 600)
      await login(form.email, form.password)
      setPhase('SUCCESS')
      setTimeout(() => navigate('/about'), 500)
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(detail || 'AUTHENTICATION FAILED')
      setPhase('IDLE')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="w-full" style={{ animation: 'slideInFromBottom 0.4s ease-out' }}>

      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{
          fontFamily: 'Share Tech Mono, monospace',
          fontSize: '0.65rem',
          letterSpacing: '0.25em',
          color: 'rgba(170,0,255,0.5)',
          marginBottom: '0.5rem',
          textTransform: 'uppercase',
        }}>
          NEURAL AUTH PROTOCOL
        </div>
        <h2 style={{
          fontFamily: 'Orbitron, monospace',
          fontSize: 'clamp(1.4rem, 5.5vw, 1.8rem)',
          fontWeight: 900,
          color: '#f0ccff',
          letterSpacing: '0.1em',
          margin: 0,
          textShadow: '0 0 20px rgba(170,0,255,0.3)',
          lineHeight: 1.05,
        }}>
          ACCESS<br />
          <span style={{ color: '#bf00ff', textShadow: '0 0 20px rgba(170,0,255,0.6)' }}>
            TERMINAL
          </span>
        </h2>
      </div>

      {/* Phase indicator */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '0.5rem',
        marginBottom: '1.5rem',
        padding: '0.5rem 0.75rem',
        background: 'rgba(170,0,255,0.05)',
        border: '1px solid rgba(170,0,255,0.15)',
      }}>
        <div style={{
          width: 6, height: 6, borderRadius: '50%',
          background: phase === 'SUCCESS' ? '#00ff88' : '#bf00ff',
          boxShadow: phase === 'SUCCESS'
            ? '0 0 6px rgba(0,255,136,0.8)'
            : '0 0 6px rgba(170,0,255,0.8)',
          animation: loading ? 'statusPulse 0.8s ease-in-out infinite' : 'statusPulse 2s ease-in-out infinite',
        }} />
        <span style={{
          fontFamily: 'Share Tech Mono, monospace',
          fontSize: 'clamp(0.55rem, 2.4vw, 0.65rem)',
          letterSpacing: '0.15em',
          color: phase === 'SUCCESS' ? '#00ff88' : 'rgba(170,0,255,0.7)',
        }}>
          {phase === 'IDLE'           && 'AWAITING CREDENTIALS'}
          {phase === 'SCANNING'       && 'SCANNING IDENTITY...'}
          {phase === 'AUTHENTICATING' && 'AUTHENTICATING...'}
          {phase === 'SUCCESS'        && 'ACCESS GRANTED'}
        </span>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

        {/* Email field */}
        <div>
          <label style={{
            display: 'block',
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.65rem',
            letterSpacing: '0.2em',
            color: 'rgba(170,0,255,0.6)',
            marginBottom: '0.4rem',
            textTransform: 'uppercase',
          }}>
            IDENTITY / EMAIL
          </label>
          <input
            type="email"
            name="email"
            value={form.email}
            onChange={handleChange}
            placeholder="user@domain.net"
            disabled={loading}
            autoComplete="email"
            style={{
              width: '100%',
              padding: '0.65rem 0.9rem',
              background: 'rgba(45,0,87,0.3)',
              border: `1px solid ${error ? 'rgba(255,0,100,0.6)' : 'rgba(170,0,255,0.3)'}`,
              color: '#f0ccff',
              fontFamily: 'Share Tech Mono, monospace',
              fontSize: 'clamp(0.78rem, 2.6vw, 0.85rem)',
              outline: 'none',
              transition: 'border-color 0.2s, box-shadow 0.2s',
            }}
            onFocus={(e) => {
              e.target.style.borderColor = 'rgba(170,0,255,0.8)'
              e.target.style.boxShadow = '0 0 12px rgba(170,0,255,0.3)'
            }}
            onBlur={(e) => {
              e.target.style.borderColor = error ? 'rgba(255,0,100,0.6)' : 'rgba(170,0,255,0.3)'
              e.target.style.boxShadow = 'none'
            }}
          />
        </div>

        {/* Password field */}
        <div>
          <label style={{
            display: 'block',
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.65rem',
            letterSpacing: '0.2em',
            color: 'rgba(170,0,255,0.6)',
            marginBottom: '0.4rem',
            textTransform: 'uppercase',
          }}>
            CIPHER / PASSWORD
          </label>
          <input
            type="password"
            name="password"
            value={form.password}
            onChange={handleChange}
            placeholder="••••••••••••"
            disabled={loading}
            autoComplete="current-password"
            style={{
              width: '100%',
              padding: '0.65rem 0.9rem',
              background: 'rgba(45,0,87,0.3)',
              border: `1px solid ${error ? 'rgba(255,0,100,0.6)' : 'rgba(170,0,255,0.3)'}`,
              color: '#f0ccff',
              fontFamily: 'Share Tech Mono, monospace',
              fontSize: 'clamp(0.78rem, 2.6vw, 0.85rem)',
              outline: 'none',
              transition: 'border-color 0.2s, box-shadow 0.2s',
            }}
            onFocus={(e) => {
              e.target.style.borderColor = 'rgba(170,0,255,0.8)'
              e.target.style.boxShadow = '0 0 12px rgba(170,0,255,0.3)'
            }}
            onBlur={(e) => {
              e.target.style.borderColor = error ? 'rgba(255,0,100,0.6)' : 'rgba(170,0,255,0.3)'
              e.target.style.boxShadow = 'none'
            }}
          />
        </div>

        {/* Error message */}
        {error && (
          <div style={{
            padding: '0.6rem 0.75rem',
            background: 'rgba(255,0,80,0.08)',
            border: '1px solid rgba(255,0,80,0.3)',
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.7rem',
            letterSpacing: '0.1em',
            color: '#ff4466',
            animation: 'fadeIn 0.2s ease-out',
          }}>
            ERROR / {error}
          </div>
        )}

        {/* Submit button */}
        <button
          type="submit"
          disabled={loading}
          style={{
            marginTop: '0.5rem',
            width: '100%',
            padding: '0.75rem',
            background: loading
              ? 'rgba(170,0,255,0.15)'
              : 'linear-gradient(135deg, rgba(102,0,179,0.8), rgba(170,0,255,0.8))',
            border: '1px solid rgba(170,0,255,0.6)',
            color: loading ? 'rgba(240,204,255,0.4)' : '#f0ccff',
            fontFamily: 'Orbitron, monospace',
            fontSize: 'clamp(0.6rem, 2.4vw, 0.7rem)',
            fontWeight: 700,
            letterSpacing: '0.25em',
            textTransform: 'uppercase',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
            boxShadow: loading ? 'none' : '0 0 20px rgba(170,0,255,0.25)',
            position: 'relative',
            overflow: 'hidden',
          }}
          onMouseEnter={(e) => {
            if (!loading) {
              e.target.style.boxShadow = '0 0 30px rgba(170,0,255,0.5), 0 0 60px rgba(170,0,255,0.2)'
              e.target.style.borderColor = 'rgba(204,68,255,0.9)'
            }
          }}
          onMouseLeave={(e) => {
            if (!loading) {
              e.target.style.boxShadow = '0 0 20px rgba(170,0,255,0.25)'
              e.target.style.borderColor = 'rgba(170,0,255,0.6)'
            }
          }}
        >
          {loading ? 'PROCESSING...' : 'INITIATE ACCESS'}
        </button>
      </form>

      {/* Divider */}
      <div style={{
        height: '1px',
        background: 'linear-gradient(90deg, transparent, rgba(170,0,255,0.3), transparent)',
        margin: '1.5rem 0',
      }} />

      {/* Register link */}
      <div style={{
        textAlign: 'center',
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.7rem',
        letterSpacing: '0.1em',
        color: 'rgba(170,0,255,0.5)',
      }}>
        NO IDENTITY REGISTERED?{' '}
        <Link
          to="/register"
          style={{
            color: '#cc44ff',
            textDecoration: 'none',
            textShadow: '0 0 8px rgba(170,0,255,0.4)',
            transition: 'color 0.2s, text-shadow 0.2s',
          }}
          onMouseEnter={(e) => {
            e.target.style.color = '#bf00ff'
            e.target.style.textShadow = '0 0 12px rgba(170,0,255,0.7)'
          }}
          onMouseLeave={(e) => {
            e.target.style.color = '#cc44ff'
            e.target.style.textShadow = '0 0 8px rgba(170,0,255,0.4)'
          }}
        >
          REGISTER NODE
        </Link>
      </div>

      <style>{`
        @keyframes slideInFromBottom {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        @keyframes statusPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.4; transform: scale(0.85); }
        }
      `}</style>
    </div>
  )
}

export default Login
