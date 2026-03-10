import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { registerUser } from '../api/auth'

// -----------------------------------------------------------------------------
// LISTS & CONFIG
// -----------------------------------------------------------------------------
const COUNTRIES = [
  "United States", "Canada", "United Kingdom", "Australia", "Germany",
  "France", "Italy", "Spain", "Japan", "China", "India", "Brazil",
  "Mexico", "Russia", "South Africa", "Argentina", "Chile", "Colombia",
  "South Korea", "Singapore", "Australia", "New Zealand", "Netherlands"
].sort();

const FIELDS = [
  {
    name: 'full_name',
    label: 'OPERATOR NAME',
    placeholder: 'John Doe',
    type: 'text',
    autoComplete: 'name',
  },
  {
    name: 'email',
    label: 'IDENTITY / EMAIL',
    placeholder: 'user@domain.net',
    type: 'email',
    autoComplete: 'email',
  },
  {
    name: 'phone_number',
    label: 'PHONE',
    placeholder: '+11234567890',
    type: 'tel',
    autoComplete: 'tel',
  },
  {
    name: 'country',
    label: 'COUNTRY',
    placeholder: 'Select Country',
    type: 'select',
    options: COUNTRIES,
    autoComplete: 'off',
  },
  {
    name: 'age',
    label: 'AGE / CYCLES',
    placeholder: '25',
    type: 'number',
    autoComplete: 'off',
    min: 1,
    max: 120,
  },
  {
    name: 'gender',
    label: 'GENDER',
    placeholder: 'Select Gender',
    type: 'select',
    options: [
      { label: 'Male', value: 'M' },
      { label: 'Female', value: 'F' }
    ],
    autoComplete: 'off',
  },
  {
    name: 'password',
    label: 'CIPHER / PASSWORD',
    placeholder: '••••••••••••',
    type: 'password',
    autoComplete: 'new-password',
  },
  {
    name: 'confirm_password',
    label: 'CONFIRM CIPHER',
    placeholder: '••••••••••••',
    type: 'password',
    autoComplete: 'new-password',
  },
]

const Register = () => {
  const navigate = useNavigate()
  const { login } = useAuth()

  const [form, setForm] = useState({
    full_name: '',
    email: '',
    phone_number: '',
    country: '',
    age: '',
    gender: '',
    password: '',
    confirm_password: '',
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [phase, setPhase] = useState('IDLE')
  const [focusedField, setFocusedField] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setForm((prev) => ({ ...prev, [name]: value }))
    if (error) setError(null)
  }

  const validate = () => {
    if (!form.full_name.trim()) return 'OPERATOR NAME REQUIRED'
    if (!form.email.trim()) return 'IDENTITY EMAIL REQUIRED'
    if (!form.age || form.age < 1) return 'VALID AGE REQUIRED'
    if (!form.gender) return 'GENDER REQUIRED'
    if (!form.country) return 'COUNTRY REQUIRED'
    if (form.password.length < 8) return 'CIPHER MUST BE 8+ CHARACTERS'
    if (form.password !== form.confirm_password) return 'CIPHERS DO NOT MATCH'
    return null
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    const validationError = validate()
    if (validationError) {
      setError(validationError)
      return
    }

    setLoading(true)
    setError(null)
    setPhase('REGISTERING')

    try {
      await registerUser({
        full_name: form.full_name.trim(),
        email: form.email.trim(),
        phone_number: form.phone_number.trim(),
        country: form.country,
        gender: form.gender,
        age: parseInt(form.age),
        password: form.password,
      })

      setPhase('AUTHENTICATING')
      await login(form.email.trim(), form.password)

      setPhase('SUCCESS')
      setTimeout(() => navigate('/about'), 500)
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(detail || 'REGISTRATION FAILED')
      setPhase('IDLE')
    } finally {
      setLoading(false)
    }
  }

  const getPhaseLabel = () => {
    switch (phase) {
      case 'REGISTERING': return 'REGISTERING NODE...'
      case 'AUTHENTICATING': return 'AUTHENTICATING...'
      case 'SUCCESS': return 'NODE REGISTERED'
      default: return 'AWAITING INPUT'
    }
  }

  return (
    <div style={{ width: '100%', animation: 'slideInFromBottom 0.4s ease-out' }}>

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
          NODE REGISTRATION PROTOCOL
        </div>
        <h2 style={{
          fontFamily: 'Orbitron, monospace',
          fontSize: '1.8rem',
          fontWeight: 900,
          color: '#f0ccff',
          letterSpacing: '0.1em',
          margin: 0,
          textShadow: '0 0 20px rgba(170,0,255,0.3)',
        }}>
          NEW<br />
          <span style={{ color: '#bf00ff', textShadow: '0 0 20px rgba(170,0,255,0.6)' }}>
            OPERATOR
          </span>
        </h2>
      </div>

      {/* Phase indicator */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
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
          animation: loading
            ? 'statusPulse 0.8s ease-in-out infinite'
            : 'statusPulse 2s ease-in-out infinite',
        }} />
        <span style={{
          fontFamily: 'Share Tech Mono, monospace',
          fontSize: '0.65rem',
          letterSpacing: '0.15em',
          color: phase === 'SUCCESS' ? '#00ff88' : 'rgba(170,0,255,0.7)',
        }}>
          {getPhaseLabel()}
        </span>
      </div>

      {/* Form */}
      <form
        onSubmit={handleSubmit}
        style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}
      >
        {FIELDS.map((field) => (
          <div key={field.name}>
            <label style={{
              display: 'block',
              fontFamily: 'Share Tech Mono, monospace',
              fontSize: '0.65rem',
              letterSpacing: '0.2em',
              color: focusedField === field.name
                ? 'rgba(204,68,255,0.9)'
                : 'rgba(170,0,255,0.6)',
              marginBottom: '0.35rem',
              textTransform: 'uppercase',
              transition: 'color 0.2s',
            }}>
              {field.label}
            </label>
            {field.type === 'select' ? (
              <select
                name={field.name}
                value={form[field.name]}
                onChange={handleChange}
                disabled={loading}
                onFocus={() => setFocusedField(field.name)}
                onBlur={() => setFocusedField(null)}
                style={{
                  width: '100%',
                  padding: '0.7rem 1rem',
                  background: focusedField === field.name
                    ? 'rgba(45,0,87,0.5)'
                    : 'rgba(45,0,87,0.3)',
                  border: `1px solid ${error && !form[field.name]
                    ? 'rgba(255,0,100,0.5)'
                    : focusedField === field.name
                      ? 'rgba(204,68,255,0.8)'
                      : 'rgba(170,0,255,0.3)'
                    }`,
                  color: form[field.name] ? '#f0ccff' : 'rgba(240,204,255,0.4)',
                  fontFamily: 'Share Tech Mono, monospace',
                  fontSize: '0.85rem',
                  outline: 'none',
                  appearance: 'none',
                  WebkitAppearance: 'none',
                  cursor: 'pointer',
                  transition: 'border-color 0.2s, box-shadow 0.2s, background 0.2s',
                  boxShadow: focusedField === field.name
                    ? '0 0 12px rgba(170,0,255,0.25)'
                    : 'none',
                }}
              >
                <option value="" disabled hidden>{field.placeholder}</option>
                {field.options.map((opt) => (
                  <option
                    key={typeof opt === 'string' ? opt : opt.value}
                    value={typeof opt === 'string' ? opt : opt.value}
                    style={{ background: '#1a0033', color: '#f0ccff' }}
                  >
                    {typeof opt === 'string' ? opt : opt.label}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type={field.type}
                name={field.name}
                value={form[field.name]}
                onChange={handleChange}
                placeholder={field.placeholder}
                disabled={loading}
                autoComplete={field.autoComplete}
                min={field.min}
                max={field.max}
                style={{
                  width: '100%',
                  padding: '0.7rem 1rem',
                  background: focusedField === field.name
                    ? 'rgba(45,0,87,0.5)'
                    : 'rgba(45,0,87,0.3)',
                  border: `1px solid ${error && !form[field.name]
                    ? 'rgba(255,0,100,0.5)'
                    : focusedField === field.name
                      ? 'rgba(204,68,255,0.8)'
                      : 'rgba(170,0,255,0.3)'
                    }`,
                  color: '#f0ccff',
                  fontFamily: 'Share Tech Mono, monospace',
                  fontSize: '0.85rem',
                  outline: 'none',
                  transition: 'border-color 0.2s, box-shadow 0.2s, background 0.2s',
                  boxShadow: focusedField === field.name
                    ? '0 0 12px rgba(170,0,255,0.25)'
                    : 'none',
                }}
                onFocus={() => setFocusedField(field.name)}
                onBlur={() => setFocusedField(null)}
              />
            )}
          </div>
        ))}

        {/* Password strength indicator */}
        {form.password.length > 0 && (
          <div style={{ display: 'flex', gap: '4px', marginTop: '-0.5rem' }}>
            {[1, 2, 3, 4].map((level) => {
              const strength = Math.min(4, Math.floor(form.password.length / 3))
              const active = level <= strength
              return (
                <div key={level} style={{
                  flex: 1,
                  height: '2px',
                  background: active
                    ? level <= 1 ? 'rgba(255,0,80,0.8)'
                      : level <= 2 ? 'rgba(255,140,0,0.8)'
                        : level <= 3 ? 'rgba(170,0,255,0.8)'
                          : 'rgba(0,255,136,0.8)'
                    : 'rgba(170,0,255,0.15)',
                  boxShadow: active ? '0 0 4px currentColor' : 'none',
                  transition: 'background 0.3s',
                }} />
              )
            })}
          </div>
        )}

        {/* Error */}
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

        {/* Submit */}
        <button
          type="submit"
          disabled={loading}
          style={{
            marginTop: '0.5rem',
            width: '100%',
            padding: '0.85rem',
            background: loading
              ? 'rgba(170,0,255,0.15)'
              : 'linear-gradient(135deg, rgba(102,0,179,0.8), rgba(170,0,255,0.8))',
            border: '1px solid rgba(170,0,255,0.6)',
            color: loading ? 'rgba(240,204,255,0.4)' : '#f0ccff',
            fontFamily: 'Orbitron, monospace',
            fontSize: '0.7rem',
            fontWeight: 700,
            letterSpacing: '0.25em',
            textTransform: 'uppercase',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
            boxShadow: loading ? 'none' : '0 0 20px rgba(170,0,255,0.25)',
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
          {loading ? 'PROCESSING...' : 'REGISTER NODE'}
        </button>
      </form>

      {/* Divider */}
      <div style={{
        height: '1px',
        background: 'linear-gradient(90deg, transparent, rgba(170,0,255,0.3), transparent)',
        margin: '1.5rem 0',
      }} />

      {/* Login link */}
      <div style={{
        textAlign: 'center',
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.7rem',
        letterSpacing: '0.1em',
        color: 'rgba(170,0,255,0.5)',
      }}>
        ALREADY REGISTERED?{' '}
        <Link
          to="/login"
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
          ACCESS TERMINAL
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

export default Register
