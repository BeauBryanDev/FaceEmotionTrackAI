import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, Cpu, Database, Terminal, Zap, RefreshCw, User, Shield } from 'lucide-react'
import { getEmotionSummary } from '../api/emotions'
import { useAuth } from '../context/AuthContext'
import MainIcon from '../assets/emotitrackincon.jpg'

// -----------------------------------------------------------------------------
// EMOTION COLOR MAP
// Each emotion gets a distinct color within the purple/violet palette
// -----------------------------------------------------------------------------
const EMOTION_COLORS = {
  Happiness: { bar: '#00ff88', glow: 'rgba(0,255,136,0.6)' },
  Neutral: { bar: '#cc44ff', glow: 'rgba(204,68,255,0.6)' },
  Sadness: { bar: '#4488ff', glow: 'rgba(68,136,255,0.6)' },
  Anger: { bar: '#ff4466', glow: 'rgba(255,68,102,0.6)' },
  Fear: { bar: '#ff9900', glow: 'rgba(255,153,0,0.6)' },
  Surprise: { bar: '#ff44cc', glow: 'rgba(255,68,204,0.6)' },
  Disgust: { bar: '#aa44ff', glow: 'rgba(170,68,255,0.6)' },
  Contempt: { bar: '#8800e6', glow: 'rgba(136,0,230,0.6)' },
}

const getEmotionColor = (emotion) =>
  EMOTION_COLORS[emotion] || { bar: '#bf00ff', glow: 'rgba(170,0,255,0.6)' }

// -----------------------------------------------------------------------------
// METRIC CARD
// Reusable card with cyberpunk corner decorations and glow
// -----------------------------------------------------------------------------
const MetricCard = ({ label, value, icon, accent = false, children }) => (
  <div style={{
    background: accent ? 'rgba(45,0,87,0.4)' : 'rgba(19,0,32,0.9)',
    border: `1px solid ${accent ? 'rgba(170,0,255,0.5)' : 'rgba(170,0,255,0.2)'}`,
    boxShadow: accent ? '0 0 20px rgba(170,0,255,0.15)' : 'none',
    padding: '1.5rem',
    position: 'relative',
    overflow: 'hidden',
    animation: 'fadeSlideIn 0.4s ease-out both',
  }}>
    {/* Corner decorations */}
    <div style={{
      position: 'absolute', top: 0, left: 0, width: 12, height: 12,
      borderTop: `1px solid ${accent ? '#bf00ff' : 'rgba(170,0,255,0.4)'}`,
      borderLeft: `1px solid ${accent ? '#bf00ff' : 'rgba(170,0,255,0.4)'}`
    }} />
    <div style={{
      position: 'absolute', bottom: 0, right: 0, width: 12, height: 12,
      borderBottom: `1px solid ${accent ? '#bf00ff' : 'rgba(170,0,255,0.4)'}`,
      borderRight: `1px solid ${accent ? '#bf00ff' : 'rgba(170,0,255,0.4)'}`
    }} />

    {/* Top shimmer line */}
    {accent && (
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '1px',
        background: 'linear-gradient(90deg, transparent, #bf00ff, transparent)',
      }} />
    )}

    {/* Background glow blob */}
    <div style={{
      position: 'absolute', top: '-20px', right: '-20px',
      width: '80px', height: '80px',
      background: accent ? 'rgba(170,0,255,0.08)' : 'rgba(170,0,255,0.03)',
      borderRadius: '50%',
      filter: 'blur(20px)',
      pointerEvents: 'none',
    }} />

    <div style={{
      fontFamily: 'Share Tech Mono, monospace',
      fontSize: '0.6rem',
      letterSpacing: '0.2em',
      color: 'rgba(170,0,255,0.5)',
      marginBottom: '0.75rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.4rem',
    }}>
      {icon}
      {label}
    </div>

    {value !== undefined && (
      <div style={{
        fontFamily: 'Orbitron, monospace',
        fontSize: '2.5rem',
        fontWeight: 900,
        color: accent ? '#bf00ff' : '#f0ccff',
        textShadow: accent ? '0 0 20px rgba(170,0,255,0.5)' : 'none',
        lineHeight: 1,
        letterSpacing: '-0.02em',
      }}>
        {value}
      </div>
    )}

    {children}
  </div>
)

// -----------------------------------------------------------------------------
// EMOTION BAR
// Animated progress bar with per-emotion color
// -----------------------------------------------------------------------------
const EmotionBar = ({ stat, index, animate }) => {
  const { bar, glow } = getEmotionColor(stat.emotion)

  return (
    <div style={{
      animation: `fadeSlideIn 0.4s ease-out ${index * 0.06}s both`,
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'baseline',
        marginBottom: '0.35rem',
      }}>
        <span style={{
          fontFamily: 'Share Tech Mono, monospace',
          fontSize: '0.72rem',
          letterSpacing: '0.12em',
          color: '#f0ccff',
        }}>
          {stat.emotion.toUpperCase()}
        </span>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'baseline' }}>
          <span style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.6rem',
            color: 'rgba(170,0,255,0.45)',
          }}>
            {stat.count} scans
          </span>
          <span style={{
            fontFamily: 'Orbitron, monospace',
            fontSize: '0.75rem',
            fontWeight: 700,
            color: bar,
            textShadow: `0 0 8px ${glow}`,
          }}>
            {stat.percentage.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Bar track */}
      <div style={{
        width: '100%',
        height: '3px',
        background: 'rgba(170,0,255,0.1)',
        border: '1px solid rgba(170,0,255,0.15)',
        overflow: 'hidden',
        position: 'relative',
      }}>
        <div style={{
          height: '100%',
          background: bar,
          boxShadow: `0 0 8px ${glow}`,
          width: animate ? `${stat.percentage}%` : '0%',
          transition: `width 1s cubic-bezier(0.22, 1, 0.36, 1) ${index * 0.08}s`,
        }} />
      </div>

      {/* Confidence */}
      <div style={{
        marginTop: '0.25rem',
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.55rem',
        color: 'rgba(170,0,255,0.35)',
        letterSpacing: '0.1em',
      }}>
        AVG CONFIDENCE: {(stat.avg_confidence * 100).toFixed(1)}%
      </div>
    </div>
  )
}

// -----------------------------------------------------------------------------
// SESSION INFO PANEL
// Shows authenticated user details and biometric enrollment status
// -----------------------------------------------------------------------------
const SessionPanel = ({ user }) => (
  <div style={{
    background: 'rgba(19,0,32,0.9)',
    border: '1px solid rgba(170,0,255,0.2)',
    padding: '1.5rem',
    position: 'relative',
    animation: 'fadeSlideIn 0.4s ease-out 0.2s both',
  }}>
    <div style={{
      position: 'absolute', top: 0, left: 0, width: 12, height: 12,
      borderTop: '1px solid rgba(170,0,255,0.4)',
      borderLeft: '1px solid rgba(170,0,255,0.4)'
    }} />
    <div style={{
      position: 'absolute', bottom: 0, right: 0, width: 12, height: 12,
      borderBottom: '1px solid rgba(170,0,255,0.4)',
      borderRight: '1px solid rgba(170,0,255,0.4)'
    }} />

    <div style={{
      fontFamily: 'Share Tech Mono, monospace',
      fontSize: '0.85rem',
      letterSpacing: '0.2em',
      color: 'rgba(212, 177, 231, 0.5)',
      marginBottom: '1rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.4rem',
    }}>
      <User size={12} /> SESSION DATA
    </div>

    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
      {[
        { label: 'OPERATOR', value: user?.full_name || 'UNKNOWN' },
        { label: 'NODE ID', value: `#${String(user?.id || '000').padStart(4, '0')}` },
        { label: 'AGE', value: user?.age ? `${user.age} CYCLES` : 'N/A' },
        {
          label: 'BIOMETRICS',
          value: user?.face_embedding ? 'ENROLLED' : 'NOT ENROLLED',
          valueColor: user?.face_embedding ? '#00ff88' : 'rgba(255,68,102,0.8)',
          valueGlow: user?.face_embedding ? '0 0 6px rgba(0,255,136,0.5)' : 'none',
        },
      ].map(({ label, value, valueColor, valueGlow }) => (
        <div key={label} style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          paddingBottom: '0.5rem',
          borderBottom: '1px solid rgba(203, 129, 238, 0.08)',
        }}>
          <span style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.76rem',
            letterSpacing: '0.12em',
            color: 'rgba(170,0,255,0.4)',
          }}>
            {label}
          </span>
          <span style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.76rem',
            letterSpacing: '0.05em',
            color: valueColor || '#cc44ff',
            textShadow: valueGlow || 'none',
          }}>
            {value}
          </span>
        </div>
      ))}
    </div>
  </div>
)

// -----------------------------------------------------------------------------
// LOADING STATE
// -----------------------------------------------------------------------------
const LoadingState = () => (
  <div style={{
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '300px',
    gap: '1rem',
  }}>
    <div style={{
      width: '120px',
      height: '2px',
      background: 'rgba(170,0,255,0.2)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute',
        height: '100%',
        width: '40%',
        background: 'linear-gradient(90deg, transparent, #bf00ff, transparent)',
        animation: 'scanLine 1.2s linear infinite',
      }} />
    </div>
    <span style={{
      fontFamily: 'Share Tech Mono, monospace',
      fontSize: '0.7rem',
      letterSpacing: '0.2em',
      color: 'rgba(170,0,255,0.5)',
      animation: 'pulse 1.5s ease-in-out infinite',
    }}>
      AGGREGATING NEURAL DATA...
    </span>
  </div>
)

// -----------------------------------------------------------------------------
// ERROR STATE
// -----------------------------------------------------------------------------
const ErrorState = ({ message, onRetry }) => (
  <div style={{
    padding: '1.25rem',
    background: 'rgba(255,0,80,0.06)',
    border: '1px solid rgba(255,0,80,0.3)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '1rem',
    animation: 'fadeSlideIn 0.3s ease-out',
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
      <Zap size={16} color="#ff4466" style={{ flexShrink: 0 }} />
      <span style={{
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.7rem',
        letterSpacing: '0.1em',
        color: '#ff4466',
      }}>
        {message}
      </span>
    </div>
    <button
      onClick={onRetry}
      style={{
        background: 'transparent',
        border: '1px solid rgba(255,0,80,0.4)',
        color: 'rgba(255,68,102,0.8)',
        fontFamily: 'Share Tech Mono, monospace',
        fontSize: '0.6rem',
        letterSpacing: '0.1em',
        padding: '0.4rem 0.75rem',
        cursor: 'pointer',
        flexShrink: 0,
        transition: 'all 0.2s',
      }}
    >
      RETRY
    </button>
  </div>
)

// -----------------------------------------------------------------------------
// EMPTY STATE
// -----------------------------------------------------------------------------
const EmptyState = ({ onNavigate }) => (
  <div style={{
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '200px',
    gap: '1rem',
    border: '1px dashed rgba(170,0,255,0.2)',
    padding: '2rem',
  }}>
    <Shield size={32} color="rgba(170,0,255,0.3)" />
    <span style={{
      fontFamily: 'Share Tech Mono, monospace',
      fontSize: '0.7rem',
      letterSpacing: '0.15em',
      color: 'rgba(170,0,255,0.4)',
      textAlign: 'center',
    }}>
      [ INSUFFICIENT DATA FOR DISTRIBUTION ANALYSIS ]
    </span>
    <button
      onClick={onNavigate}
      style={{
        background: 'transparent',
        border: '1px solid rgba(170,0,255,0.3)',
        color: 'rgba(170,0,255,0.6)',
        fontFamily: 'Orbitron, monospace',
        fontSize: '0.6rem',
        fontWeight: 700,
        letterSpacing: '0.15em',
        padding: '0.6rem 1.25rem',
        cursor: 'pointer',
        transition: 'all 0.2s',
      }}
      onMouseEnter={(e) => {
        e.target.style.background = 'rgba(170,0,255,0.1)'
        e.target.style.borderColor = 'rgba(170,0,255,0.6)'
        e.target.style.color = '#cc44ff'
        e.target.style.boxShadow = '0 0 12px rgba(170,0,255,0.2)'
      }}
      onMouseLeave={(e) => {
        e.target.style.background = 'transparent'
        e.target.style.borderColor = 'rgba(170,0,255,0.3)'
        e.target.style.color = 'rgba(170,0,255,0.6)'
        e.target.style.boxShadow = 'none'
      }}
    >
      INITIATE LIVE SCAN
    </button>
  </div>
)

// -----------------------------------------------------------------------------
// DASHBOARD
// -----------------------------------------------------------------------------
const Dashboard = () => {
  const { user } = useAuth()
  const navigate = useNavigate()

  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [animate, setAnimate] = useState(false)
  const [lastSync, setLastSync] = useState(null)

  const fetchSummary = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await getEmotionSummary()
      setSummary(data)
      setLastSync(new Date())
      setTimeout(() => setAnimate(true), 100)
    } catch (err) {
      console.error('Dashboard fetch error:', err)
      setError('UNABLE TO ESTABLISH UPLINK WITH MAIN NEURAL CORE')
    } finally {
      setLoading(false)
    }
  }, [])

  // Initial fetch
  useEffect(() => {
    fetchSummary()
  }, [fetchSummary])

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimate(false)
      fetchSummary()
    }, 30000)
    return () => clearInterval(interval)
  }, [fetchSummary])

  const handleRefresh = () => {
    setAnimate(false)
    fetchSummary()
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

      {/* Page header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-end',
        paddingBottom: '1.25rem',
        borderBottom: '1px solid rgba(170,0,255,0.15)',
        animation: 'fadeSlideIn 0.3s ease-out',
      }}>
        <div>
          <div style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.6rem',
            letterSpacing: '0.25em',
            color: 'rgba(170,0,255,0.4)',
            marginBottom: '0.4rem',
          }}>
            SYSTEM / TELEMETRY
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
            <Terminal size={22} color="#bf00ff" />
            COMMAND CENTER
          </h1>
        </div>

        {/* Refresh control */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.25rem' }}>
          <button
            onClick={handleRefresh}
            disabled={loading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: 'transparent',
              border: '1px solid rgba(170,0,255,0.3)',
              color: 'rgba(170,0,255,0.6)',
              fontFamily: 'Share Tech Mono, monospace',
              fontSize: '0.6rem',
              letterSpacing: '0.15em',
              padding: '0.5rem 0.85rem',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.5 : 1,
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              if (!loading) e.currentTarget.style.borderColor = 'rgba(170,0,255,0.6)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'rgba(170,0,255,0.3)'
            }}
          >
            <RefreshCw
              size={12}
              style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }}
            />
            REFRESH
          </button>
          {lastSync && (
            <span style={{
              fontFamily: 'Share Tech Mono, monospace',
              fontSize: '0.5rem',
              color: 'rgba(170,0,255,0.25)',
              letterSpacing: '0.1em',
            }}>
              LAST SYNC {lastSync.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Error */}
      {error && <ErrorState message={error} onRetry={handleRefresh} />}

      {/* Loading */}
      {loading && !error && <LoadingState />}

      {/* Main content */}
      {!loading && !error && summary && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 2fr',
          gridTemplateRows: 'auto auto',
          gap: '1.25rem',
        }}>

          {/* Total scans */}
          <MetricCard
            label="TOTAL BIOMETRIC SCANS"
            value={summary.total_detections ?? 0}
            icon={<Database size={12} />}
          />

          {/* Dominant emotion */}
          <MetricCard
            label="PRIMARY NEURAL STATE"
            value={summary.dominant_emotion || 'N/A'}
            icon={<Cpu size={12} />}
            accent
          />

          {/* Session info - spans right column first row */}
          <div style={{ gridRow: '1 / 3' }}>
            <SessionPanel user={user} />
          </div>

          {/* Quick actions */}
          <div style={{
            gridColumn: '1 / 3',
            display: 'flex',
            gap: '0.75rem',
            animation: 'fadeSlideIn 0.4s ease-out 0.15s both',
          }}>
            <button
              onClick={() => navigate('/dashboard')}
              style={{
                flex: 1,
                padding: '0.75rem',
                background: 'linear-gradient(135deg, rgba(102,0,179,0.6), rgba(170,0,255,0.6))',
                border: '1px solid rgba(170,0,255,0.5)',
                color: '#f0ccff',
                fontFamily: 'Orbitron, monospace',
                fontSize: '0.6rem',
                fontWeight: 700,
                letterSpacing: '0.15em',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                transition: 'all 0.2s',
                boxShadow: '0 0 16px rgba(170,0,255,0.2)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.boxShadow = '0 0 24px rgba(170,0,255,0.4)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow = '0 0 16px rgba(170,0,255,0.2)'
              }}
            >
              <Activity size={14} /> INITIATE LIVE SCAN
            </button>
            <button
              onClick={() => navigate('/emotions')}
              style={{
                flex: 1,
                padding: '0.75rem',
                background: 'transparent',
                border: '1px solid rgba(170,0,255,0.3)',
                color: 'rgba(170,0,255,0.7)',
                fontFamily: 'Orbitron, monospace',
                fontSize: '0.6rem',
                fontWeight: 700,
                letterSpacing: '0.15em',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(170,0,255,0.08)'
                e.currentTarget.style.borderColor = 'rgba(170,0,255,0.6)'
                e.currentTarget.style.color = '#cc44ff'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent'
                e.currentTarget.style.borderColor = 'rgba(170,0,255,0.3)'
                e.currentTarget.style.color = 'rgba(170,0,255,0.7)'
              }}
            >
              ACCESS ARCHIVES
            </button>
          </div>

          {/* Emotion distribution - full width bottom row */}
          <div style={{
            gridColumn: '1 / 3',
            background: 'rgba(19,0,32,0.9)',
            border: '1px solid rgba(170,0,255,0.2)',
            padding: '1.5rem',
            position: 'relative',
            animation: 'fadeSlideIn 0.4s ease-out 0.25s both',
          }}>
            <div style={{
              position: 'absolute', top: 0, left: 0, width: 12, height: 12,
              borderTop: '1px solid rgba(170,0,255,0.4)',
              borderLeft: '1px solid rgba(170,0,255,0.4)'
            }} />
            <div style={{
              position: 'absolute', bottom: 0, right: 0, width: 12, height: 12,
              borderBottom: '1px solid rgba(170,0,255,0.4)',
              borderRight: '1px solid rgba(170,0,255,0.4)'
            }} />

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
              gap: '0.4rem',
            }}>
              <Activity size={12} /> NEURAL DISTRIBUTION MATRIX
            </div>

            {summary.emotion_stats && summary.emotion_stats.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {summary.emotion_stats.map((stat, i) => (
                  <EmotionBar
                    key={stat.emotion}
                    stat={stat}
                    index={i}
                    animate={animate}
                  />
                ))}
              </div>
            ) : (
              <EmptyState onNavigate={() => navigate('/dashboard')} />
            )}
          </div>

          {/* Cyber Icon - Right of Matrix */}
          <div style={{
            gridColumn: '3',
            background: 'rgba(19,0,32,0.9)',
            border: '1px solid rgba(170,0,255,0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            overflow: 'hidden',
            animation: 'fadeSlideIn 0.4s ease-out 0.3s both',
          }}>
            <div style={{ position: 'absolute', top: 0, left: 0, width: 6, height: 6, borderTop: '1px solid #bf00ff', borderLeft: '1px solid #bf00ff' }} />
            <div style={{ position: 'absolute', bottom: 0, right: 0, width: 6, height: 6, borderBottom: '1px solid #bf00ff', borderRight: '1px solid #bf00ff' }} />

            <div style={{
              position: 'relative',
              width: '85%',
              height: '85%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              {/* Scanline overlay */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '2px',
                background: 'rgba(191,0,255,0.4)',
                boxShadow: '0 0 10px #bf00ff',
                zIndex: 2,
                pointerEvents: 'none',
                animation: 'cyberScan 3s linear infinite'
              }} />

              <img
                src={MainIcon}
                alt="EmotiTrack Cybercore"
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  filter: 'hue-rotate(280deg) brightness(1.2) contrast(1.1)',
                  opacity: 0.8,
                  maskImage: 'linear-gradient(to bottom, black 95%, transparent)',
                  animation: 'glitchPulse 4s infinite'
                }}
              />

              {/* Text label overlay */}
              <div style={{
                position: 'absolute',
                bottom: '10%',
                left: 0,
                right: 0,
                textAlign: 'center',
                background: 'rgba(0,0,0,0.6)',
                backdropFilter: 'blur(4px)',
                color: '#f0ccff',
                fontFamily: 'Orbitron, monospace',
                fontSize: '0.5rem',
                letterSpacing: '0.3em',
                padding: '4px 0',
                borderTop: '1px solid rgba(170,0,255,0.3)',
                borderBottom: '1px solid rgba(170,0,255,0.3)',
                zIndex: 3
              }}>
                NEURAL_LINK_ESTABLISHED
              </div>
            </div>
          </div>

        </div>
      )}

      <style>{`
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes scanLine {
          0%   { left: -40%; }
          100% { left: 140%; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.4; }
        }
        @keyframes cyberScan {
          0% { top: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        @keyframes glitchPulse {
          0% { transform: scale(1); filter: hue-rotate(280deg) brightness(1.2); }
          1% { transform: translate(-2px, 1px); filter: hue-rotate(0deg) brightness(1.5); }
          2% { transform: translate(2px, -1px); }
          3% { transform: translate(0, 0); filter: hue-rotate(280deg) brightness(1.2); }
          50% { transform: scale(1.02); }
          100% { transform: scale(1); }
        }
      `}</style>
    </div>
  )
}

export default Dashboard

