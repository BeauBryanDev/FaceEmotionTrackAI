import { useState, useRef, useCallback, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { useBiometrics } from '../context/Biometrics'
import { updateCurrentUser, deleteCurrentUser } from '../api/users'
import { Shield, ShieldCheck, ShieldAlert, Camera, User, Trash2, RefreshCw } from 'lucide-react'
import BiometricVerifyModal from '../components/BiometricVerifyModal'
import CyberCalendar from '../components/CyberCalendar'

// -----------------------------------------------------------------------------
// LISTS & CONFIG
// -----------------------------------------------------------------------------
const COUNTRIES = [
  "United States", "Canada", "United Kingdom", "Australia", "Germany",
  "France", "Italy", "Spain", "Japan", "China", "India", "Brazil",
  "Mexico", "Russia", "South Africa", "Argentina", "Chile", "Colombia",
  "South Korea", "Singapore", "Australia", "New Zealand", "Netherlands"
].sort();

// -----------------------------------------------------------------------------
// SECTION CARD - reusable container with cyberpunk corner decorations
// -----------------------------------------------------------------------------
const SectionCard = ({ children, accent = false, danger = false }) => (
  <div style={{
    background: danger ? 'rgba(30,0,10,0.9)' : 'rgba(19,0,32,0.9)',
    border: `1px solid ${danger ? 'rgba(255,0,80,0.25)' : accent ? 'rgba(170,0,255,0.4)' : 'rgba(170,0,255,0.2)'}`,
    boxShadow: accent ? '0 0 20px rgba(170,0,255,0.1)' : 'none',
    padding: '1.5rem',
    position: 'relative',
  }}>
    <div style={{
      position: 'absolute', top: 0, left: 0, width: 12, height: 12,
      borderTop: `1px solid ${danger ? 'rgba(255,0,80,0.4)' : 'rgba(170,0,255,0.4)'}`,
      borderLeft: `1px solid ${danger ? 'rgba(255,0,80,0.4)' : 'rgba(170,0,255,0.4)'}`
    }} />
    <div style={{
      position: 'absolute', bottom: 0, right: 0, width: 12, height: 12,
      borderBottom: `1px solid ${danger ? 'rgba(255,0,80,0.4)' : 'rgba(170,0,255,0.4)'}`,
      borderRight: `1px solid ${danger ? 'rgba(255,0,80,0.4)' : 'rgba(170,0,255,0.4)'}`
    }} />
    {children}
  </div>
)

// -----------------------------------------------------------------------------
// FIELD - reusable labeled input
// -----------------------------------------------------------------------------
const Field = ({ label, name, type = 'text', value, onChange, placeholder = '' }) => (
  <div>
    <label style={{
      display: 'block',
      fontFamily: 'Share Tech Mono, monospace',
      fontSize: '0.6rem',
      letterSpacing: '0.2em',
      color: 'rgba(170,0,255,0.5)',
      marginBottom: '0.35rem',
      textTransform: 'uppercase',
    }}>
      {label}
    </label>
    <input
      type={type}
      name={name}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      style={{
        width: '100%',
        padding: '0.65rem 0.9rem',
        background: 'rgba(45,0,87,0.25)',
        border: '1px solid rgba(170,0,255,0.25)',
        color: '#f0ccff',
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.82rem',
        outline: 'none',
        transition: 'border-color 0.2s, box-shadow 0.2s',
      }}
      onFocus={(e) => {
        e.target.style.borderColor = 'rgba(170,0,255,0.7)'
        e.target.style.boxShadow = '0 0 10px rgba(170,0,255,0.2)'
      }}
      onBlur={(e) => {
        e.target.style.borderColor = 'rgba(170,0,255,0.25)'
        e.target.style.boxShadow = 'none'
      }}
    />
  </div>
)

const SelectField = ({ label, name, options, value, onChange, placeholder = '' }) => (
  <div>
    <label style={{
      display: 'block',
      fontFamily: 'Share Tech Mono, monospace',
      fontSize: '0.6rem',
      letterSpacing: '0.2em',
      color: 'rgba(170,0,255,0.5)',
      marginBottom: '0.35rem',
      textTransform: 'uppercase',
    }}>
      {label}
    </label>
    <select
      name={name}
      value={value}
      onChange={onChange}
      style={{
        width: '100%',
        padding: '0.65rem 0.9rem',
        background: 'rgba(45,0,87,0.25)',
        border: '1px solid rgba(170,0,255,0.25)',
        color: value ? '#f0ccff' : 'rgba(240,204,255,0.4)',
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.82rem',
        outline: 'none',
        appearance: 'none',
        WebkitAppearance: 'none',
        cursor: 'pointer',
        transition: 'border-color 0.2s, box-shadow 0.2s',
      }}
      onFocus={(e) => {
        e.target.style.borderColor = 'rgba(170,0,255,0.7)'
        e.target.style.boxShadow = '0 0 10px rgba(170,0,255,0.2)'
      }}
      onBlur={(e) => {
        e.target.style.borderColor = 'rgba(170,0,255,0.25)'
        e.target.style.boxShadow = 'none'
      }}
    >
      <option value="" disabled hidden>{placeholder}</option>
      {options.map((opt) => (
        <option
          key={typeof opt === 'string' ? opt : opt.value}
          value={typeof opt === 'string' ? opt : opt.value}
          style={{ background: '#1a0033', color: '#f0ccff' }}
        >
          {typeof opt === 'string' ? opt : opt.label}
        </option>
      ))}
    </select>
  </div>
)

// -----------------------------------------------------------------------------
// CYBER BUTTON
// -----------------------------------------------------------------------------
const CyberBtn = ({ children, onClick, disabled, variant = 'default', fullWidth = false, type = 'button' }) => {
  const colors = {
    primary: {
      bg: 'linear-gradient(135deg, rgba(102,0,179,0.7), rgba(170,0,255,0.7))',
      border: 'rgba(170,0,255,0.6)',
      color: '#f0ccff',
      hover: '0 0 20px rgba(170,0,255,0.35)',
    },
    danger: {
      bg: 'transparent',
      border: 'rgba(255,0,80,0.35)',
      color: 'rgba(255,80,120,0.7)',
      hover: '0 0 12px rgba(255,0,80,0.2)',
    },
    default: {
      bg: 'transparent',
      border: 'rgba(170,0,255,0.3)',
      color: 'rgba(170,0,255,0.7)',
      hover: '0 0 12px rgba(170,0,255,0.15)',
    },
  }
  const c = colors[variant]
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      style={{
        width: fullWidth ? '100%' : 'auto',
        padding: '0.65rem 1.25rem',
        background: disabled ? 'rgba(170,0,255,0.05)' : c.bg,
        border: `1px solid ${disabled ? 'rgba(170,0,255,0.15)' : c.border}`,
        color: disabled ? 'rgba(170,0,255,0.25)' : c.color,
        fontFamily: 'Orbitron, monospace',
        fontSize: '0.6rem',
        fontWeight: 700,
        letterSpacing: '0.2em',
        textTransform: 'uppercase',
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'all 0.2s',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '0.5rem',
      }}
      onMouseEnter={(e) => {
        if (!disabled) e.currentTarget.style.boxShadow = c.hover
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = 'none'
      }}
    >
      {children}
    </button>
  )
}
// -----------------------------------------------------------------------------
// BIOMETRIC ENROLLMENT PANEL
// Webcam preview -> capture -> enroll
// -----------------------------------------------------------------------------
const BiometricsPanel = () => {
  const { hasEmbedding, loading, enroll, remove } = useBiometrics()

  const [webcamActive, setWebcamActive] = useState(false)
  const [capturing, setCapturing] = useState(false)
  const [enrollStatus, setEnrollStatus] = useState('IDLE')
  const [statusMsg, setStatusMsg] = useState(null)

  const videoRef = useRef(null)
  const canvasRef = useRef(document.createElement('canvas'))
  const streamRef = useRef(null)

  useEffect(() => {
    if (webcamActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
    }
  }, [webcamActive])

  // Open webcam
  const openWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      })
      streamRef.current = stream
      setWebcamActive(true)   // First time we open the webcam
      setStatusMsg(null)
      // Save the stream to the video element
      if (videoRef.current) videoRef.current.srcObject = stream

    } catch {
      setStatusMsg('CAMERA ACCESS DENIED. CHECK BROWSER PERMISSIONS.')
    }
  }

  // Close webcam and release tracks
  const closeWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    setWebcamActive(false)
  }, [])

  // Capture current frame and send to backend
  const captureAndEnroll = async () => {
    const video = videoRef.current
    if (!video || video.readyState < 2) return

    setCapturing(true)
    setEnrollStatus('CAPTURING')

    const canvas = canvasRef.current
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)

    canvas.toBlob(async (blob) => {
      if (!blob) {
        setStatusMsg('FAILED TO CAPTURE FRAME.')
        setCapturing(false)
        setEnrollStatus('IDLE')
        return
      }

      const imageFile = new File([blob], 'biometric_capture.jpg', { type: 'image/jpeg' })

      setEnrollStatus('ENROLLING')
      try {
        await enroll(imageFile)
        setEnrollStatus('SUCCESS')
        setStatusMsg('BIOMETRIC TEMPLATE STORED SUCCESSFULLY.')
        closeWebcam()
        setTimeout(() => setEnrollStatus('IDLE'), 3000)
      } catch (err) {
        const detail = err.response?.data?.detail || 'ENROLLMENT FAILED.'
        setStatusMsg(`ERROR: ${detail}`)
        setEnrollStatus('ERROR')
        setTimeout(() => setEnrollStatus('IDLE'), 3000)
      } finally {
        setCapturing(false)
      }
    }, 'image/jpeg', 0.92)
  }

  const handleRemove = async () => {
    try {
      await remove()
      setStatusMsg('BIOMETRIC TEMPLATE REMOVED.')
    } catch {
      setStatusMsg('FAILED TO REMOVE BIOMETRICS.')
    }
  }

  const enrollBtnLabel = {
    IDLE: 'CAPTURE AND ENROLL',
    CAPTURING: 'CAPTURING...',
    ENROLLING: 'ENROLLING...',
    SUCCESS: 'ENROLLED',
    ERROR: 'RETRY',
  }[enrollStatus]

  const [showVerifyModal, setShowVerifyModal] = useState(false)
  const pendingPayloadRef = useRef(null)


  return (
    <SectionCard accent={hasEmbedding}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '1.25rem',
        paddingBottom: '0.75rem',
        borderBottom: '1px solid rgba(170,0,255,0.1)',
      }}>
        <div style={{
          fontFamily: 'Share Tech Mono, monospace',
          fontSize: '1.2rem',
          letterSpacing: '0.2em',
          color: 'rgba(165, 111, 172, 0.5)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
        }}>
          <Shield size={13} /> BIOMETRIC ENROLLMENT
        </div>

        {/* Enrollment status badge */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          padding: '0.3rem 0.6rem',
          background: hasEmbedding ? 'rgba(0,255,136,0.08)' : 'rgba(170,0,255,0.05)',
          border: `1px solid ${hasEmbedding ? 'rgba(0,255,136,0.3)' : 'rgba(170,0,255,0.2)'}`,
        }}>
          {hasEmbedding
            ? <ShieldCheck size={12} color="#00ff88" />
            : <ShieldAlert size={12} color="rgba(170,0,255,0.5)" />
          }
          <span style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.58rem',
            letterSpacing: '0.1em',
            color: hasEmbedding ? '#00ff88' : 'rgba(170,0,255,0.5)',
          }}>
            {hasEmbedding ? 'ENROLLED' : 'NOT ENROLLED'}
          </span>
        </div>
      </div>

      {/* Status message */}
      {statusMsg && (
        <div style={{
          marginBottom: '1rem',
          padding: '0.5rem 0.75rem',
          background: enrollStatus === 'SUCCESS'
            ? 'rgba(0,255,136,0.06)'
            : 'rgba(255,0,80,0.06)',
          border: `1px solid ${enrollStatus === 'SUCCESS'
            ? 'rgba(0,255,136,0.25)'
            : 'rgba(255,0,80,0.25)'}`,
          fontFamily: 'Share Tech Mono, monospace',
          fontSize: '0.62rem',
          letterSpacing: '0.1em',
          color: enrollStatus === 'SUCCESS' ? '#00ff88' : '#ff4466',
        }}>
          {statusMsg}
        </div>
      )}

      {/* Webcam preview */}
      {webcamActive && (
        <div style={{
          marginBottom: '1rem',
          position: 'relative',
          border: '1px solid rgba(170,0,255,0.3)',
          overflow: 'hidden',
          background: '#0d0010',
        }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ width: '100%', display: 'block' }}
          />
          {/* Scanline overlay */}
          <div style={{
            position: 'absolute', inset: 0,
            backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(170,0,255,0.03) 2px, rgba(170,0,255,0.03) 4px)',
            pointerEvents: 'none',
          }} />
          {/* Corner markers */}
          {[
            { top: 8, left: 8, borderTop: true, borderLeft: true },
            { top: 8, right: 8, borderTop: true, borderRight: true },
            { bottom: 8, left: 8, borderBottom: true, borderLeft: true },
            { bottom: 8, right: 8, borderBottom: true, borderRight: true },
          ].map((pos, i) => (
            <div key={i} style={{
              position: 'absolute',
              width: 16, height: 16,
              ...pos,
              borderTop: pos.borderTop ? '2px solid rgba(170,0,255,0.6)' : 'none',
              borderLeft: pos.borderLeft ? '2px solid rgba(170,0,255,0.6)' : 'none',
              borderBottom: pos.borderBottom ? '2px solid rgba(170,0,255,0.6)' : 'none',
              borderRight: pos.borderRight ? '2px solid rgba(170,0,255,0.6)' : 'none',
            }} />
          ))}
          {/* Label */}
          <div style={{
            position: 'absolute', bottom: 8, left: 0, right: 0,
            textAlign: 'center',
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.55rem',
            letterSpacing: '0.15em',
            color: 'rgba(170,0,255,0.6)',
          }}>
            POSITION FACE WITHIN FRAME
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
        {!webcamActive ? (
          <CyberBtn
            onClick={openWebcam}
            variant="primary"
            fullWidth
            disabled={loading}
          >
            <Camera size={13} />
            {hasEmbedding ? 'RE-ENROLL BIOMETRICS' : 'START ENROLLMENT'}
          </CyberBtn>
        ) : (
          <div style={{ display: 'flex', gap: '0.6rem' }}>
            <CyberBtn
              onClick={captureAndEnroll}
              variant="primary"
              fullWidth
              disabled={capturing || loading}
            >
              <Camera size={13} />
              {enrollBtnLabel}
            </CyberBtn>
            <CyberBtn
              onClick={closeWebcam}
              disabled={capturing}
            >
              CANCEL
            </CyberBtn>
          </div>
        )}

        {hasEmbedding && !webcamActive && (
          <CyberBtn
            onClick={handleRemove}
            variant="danger"
            fullWidth
            disabled={loading}
          >
            <Trash2 size={13} />
            REMOVE BIOMETRIC TEMPLATE
          </CyberBtn>
        )}
      </div>
    </SectionCard>
  )
}

// -----------------------------------------------------------------------------
// PROFILE PAGE
// -----------------------------------------------------------------------------
const Profile = () => {
  const { user, setUser, logout } = useAuth()
  const { hasEmbedding } = useBiometrics()

  const [form, setForm] = useState({
    full_name: user?.full_name || '',
    email: user?.email || '',
    age: user?.age || '',
    gender: user?.gender || '',
    country: user?.country || '',
    phone_number: user?.phone_number || '',
    password: '',
  })
  const [saving, setSaving] = useState(false)
  const [profileMsg, setProfileMsg] = useState(null)
  const [profileOk, setProfileOk] = useState(false)
  const [showVerifyModal, setShowVerifyModal] = useState(false)

  const pendingPayloadRef = useRef(null)

  const onChange = (e) =>
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }))

  const saveProfile = async (e) => {

    e.preventDefault()
    const payload = {
      full_name: form.full_name,
      email: form.email,
      age: form.age ? parseInt(form.age) : undefined,
      gender: form.gender || undefined,
      country: form.country || undefined,
      phone_number: form.phone_number || undefined,
      password: form.password || undefined,
    }
    if (hasEmbedding) {
      // store payload and open biometric verification modal
      pendingPayloadRef.current = payload
      setShowVerifyModal(true)
      return
    }
    // no embedding enrolled - save directly
    await executeSave(payload)
  }

  const executeSave = async (payload) => {
    setSaving(true)
    setProfileMsg(null)

    try {
      const updated = await updateCurrentUser(payload)
      setUser(updated)
      setForm((prev) => ({ ...prev, password: '' }))
      setProfileMsg('PROFILE UPDATED SUCCESSFULLY.')
      setProfileOk(true)
    } catch {
      setProfileMsg('FAILED TO UPDATE PROFILE.')
      setProfileOk(false)
    } finally {
      setSaving(false)
      setTimeout(() => setProfileMsg(null), 3000)
    }
  }

  const removeAccount = async () => {
    if (!window.confirm('CONFIRM: Delete your account permanently?')) return
    try {
      await deleteCurrentUser()
      logout()
    } catch {
      setProfileMsg('FAILED TO DELETE ACCOUNT.')
      setProfileOk(false)
    }
  }


  return (

    <>
      {showVerifyModal && (
        <BiometricVerifyModal
          onVerified={async () => {
            setShowVerifyModal(false)
            await executeSave(pendingPayloadRef.current)
          }}
          onCancel={() => setShowVerifyModal(false)}
        />
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

        {/* Page header */}
        <div style={{
          paddingBottom: '1.25rem',
          borderBottom: '1px solid rgba(170,0,255,0.15)',
        }}>
          <div style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.6rem',
            letterSpacing: '0.25em',
            color: 'rgba(170,0,255,0.4)',
            marginBottom: '0.4rem',
          }}>
            SYSTEM / OPERATOR
          </div>
          <h1 style={{
            fontFamily: 'Orbitron, monospace',
            fontSize: '1.5rem',
            fontWeight: 900,
            color: '#f0ccff',
            letterSpacing: '0.15em',
            margin: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
          }}>
            <User size={22} color="#bf00ff" />
            OPERATOR PROFILE
          </h1>
        </div>

        {/* Two column layout on large screens */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '1.5rem',
          alignItems: 'start',
        }}>

          {/* LEFT - Profile form */}
          <SectionCard>
            <div style={{
              fontFamily: 'Share Tech Mono, monospace',
              fontSize: '0.6rem',
              letterSpacing: '0.2em',
              color: 'rgba(170,0,255,0.5)',
              marginBottom: '1.25rem',
              paddingBottom: '0.75rem',
              borderBottom: '1px solid rgba(170,0,255,0.1)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}>
              <User size={13} /> IDENTITY DATA
            </div>

            {profileMsg && (
              <div style={{
                marginBottom: '1rem',
                padding: '0.5rem 0.75rem',
                background: profileOk ? 'rgba(0,255,136,0.06)' : 'rgba(255,0,80,0.06)',
                border: `1px solid ${profileOk ? 'rgba(0,255,136,0.25)' : 'rgba(255,0,80,0.25)'}`,
                fontFamily: 'Share Tech Mono, monospace',
                fontSize: '0.62rem',
                letterSpacing: '0.1em',
                color: profileOk ? '#00ff88' : '#ff4466',
              }}>
                {profileMsg}
              </div>
            )}

            <form
              onSubmit={saveProfile}
              style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}
            >
              <Field label="Full Name" name="full_name" value={form.full_name} onChange={onChange} />
              <Field label="Email" name="email" value={form.email} onChange={onChange} type="email" />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <Field label="Age" name="age" value={form.age} onChange={onChange} type="number" />
                <SelectField
                  label="Gender"
                  name="gender"
                  value={form.gender}
                  onChange={onChange}
                  placeholder="Select Gender"
                  options={[{ label: 'Male', value: 'M' }, { label: 'Female', value: 'F' }]}
                />
              </div>
              <SelectField
                label="Country"
                name="country"
                value={form.country}
                onChange={onChange}
                placeholder="Select Country"
                options={COUNTRIES}
              />
              <Field label="Phone" name="phone_number" value={form.phone_number} onChange={onChange} placeholder="Optional" />
              <Field label="New Password" name="password" value={form.password} onChange={onChange} type="password" placeholder="Leave blank to keep current" />

              <CyberBtn type="submit" variant="primary" fullWidth disabled={saving}>
                <RefreshCw size={13} style={{ animation: saving ? 'spin 1s linear infinite' : 'none' }} />
                {saving ? 'SAVING...' : 'SAVE PROFILE'}
              </CyberBtn>
            </form>
          </SectionCard>

          {/* RIGHT - Biometrics & Analytics */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <BiometricsPanel />
            <CyberCalendar />
          </div>
        </div>

        {/* DANGER ZONE */}
        <SectionCard danger>
          <div style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.6rem',
            letterSpacing: '0.2em',
            color: 'rgba(255,0,80,0.5)',
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}>
            <Trash2 size={13} /> DANGER ZONE
          </div>
          <p style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.62rem',
            color: 'rgba(255,80,120,0.5)',
            marginBottom: '1rem',
            letterSpacing: '0.05em',
          }}>
            Permanently deletes your account, all emotion records, and biometric data. This action cannot be undone.
          </p>
          <CyberBtn onClick={removeAccount} variant="danger">
            <Trash2 size={13} /> DELETE ACCOUNT
          </CyberBtn>
        </SectionCard>

        <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>
      </div>
    </>

  )
}

export default Profile
