import React from 'react';

// -----------------------------------------------------------------------------
// CYBER CALENDAR HUD
// Custom built tactical calendar for tracking temporal neural cycles
// -----------------------------------------------------------------------------
const CyberCalendar = () => {
    const now = new Date()
    const currentDay = now.getDate()
    const monthName = now.toLocaleString('default', { month: 'long' }).toUpperCase()
    const year = now.getFullYear()

    const firstDay = new Date(now.getFullYear(), now.getMonth(), 1).getDay()
    const daysInMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate()

    const days = Array.from({ length: daysInMonth }, (_, i) => i + 1)
    const blanks = Array.from({ length: firstDay }, (_, i) => i)
    const weekDays = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

    return (
        <div style={{
            background: 'rgba(19,0,32,0.95)',
            border: '1px solid rgba(170,0,255,0.25)',
            padding: '1.25rem',
            position: 'relative',
            fontFamily: 'Share Tech Mono, monospace',
            animation: 'fadeSlideIn 0.4s ease-out 0.35s both',
        }}>
            {/* HUD Decorations */}
            <div style={{ position: 'absolute', top: 0, left: 0, width: 8, height: 8, borderTop: '1px solid #bf00ff', borderLeft: '1px solid #bf00ff' }} />
            <div style={{ position: 'absolute', top: 0, right: 0, width: 8, height: 8, borderTop: '1px solid #bf00ff', borderRight: '1px solid #bf00ff' }} />
            <div style={{ position: 'absolute', bottom: 0, left: 0, width: 8, height: 8, borderBottom: '1px solid #bf00ff', borderLeft: '1px solid #bf00ff' }} />
            <div style={{ position: 'absolute', bottom: 0, right: 0, width: 8, height: 8, borderBottom: '1px solid #bf00ff', borderRight: '1px solid #bf00ff' }} />

            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '1rem',
                borderBottom: '1px solid rgba(170,0,255,0.15)',
                paddingBottom: '0.5rem'
            }}>
                <div style={{ fontSize: '0.6rem', letterSpacing: '0.2em', color: 'rgba(185, 123, 216, 0.5)' }}>
                    SYSTEM_TIME / DATE_COORD
                </div>
                <div style={{ fontSize: '0.85rem', fontWeight: 900, color: '#f0ccff', letterSpacing: '0.1em' }}>
                    {monthName} {year}
                </div>
            </div>

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(7, 1fr)',
                gap: '4px',
                textAlign: 'center'
            }}>
                {weekDays.map(wd => (
                    <div key={wd} style={{ fontSize: '0.5rem', color: 'rgba(170,0,255,0.4)', marginBottom: '4px' }}>{wd}</div>
                ))}
                {blanks.map(b => <div key={`b-${b}`} />)}
                {days.map(d => {
                    const isToday = d === currentDay
                    return (
                        <div key={d} style={{
                            fontSize: '0.65rem',
                            padding: '4px 0',
                            color: isToday ? '#bf00ff' : 'rgba(240,204,255,0.7)',
                            background: isToday ? 'rgba(170,0,255,0.15)' : 'transparent',
                            border: isToday ? '1px solid rgba(170,0,255,0.4)' : '1px solid transparent',
                            boxShadow: isToday ? '0 0 10px rgba(170,0,255,0.2)' : 'none',
                            fontWeight: isToday ? 900 : 400,
                            position: 'relative',
                            overflow: 'hidden'
                        }}>
                            {d}
                            {isToday && (
                                <div style={{
                                    position: 'absolute',
                                    top: 0, left: 0, width: '100%', height: '1px',
                                    background: '#bf00ff',
                                    animation: 'cyberScan 2s linear infinite'
                                }} />
                            )}
                        </div>
                    )
                })}
            </div>

            <div style={{
                marginTop: '0.75rem',
                fontSize: '0.5rem',
                color: 'rgba(170,0,255,0.3)',
                textAlign: 'right',
                letterSpacing: '0.1em'
            }}>
                SYNC_STATUS: STABLE_V4.2
            </div>
        </div>
    )
}

export default CyberCalendar;
