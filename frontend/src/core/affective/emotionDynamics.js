// Emotion Dynamics Engine

export function emotionalVelocity(prev, current, dt = 1) {
  const dx = current.valence - prev.valence
  const dy = current.arousal - prev.arousal

  return {
    vx: dx / dt,
    vy: dy / dt,
    magnitude: Math.sqrt(dx * dx + dy * dy) / dt
  }
}

export function emotionalAcceleration(prevVel, vel, dt = 1) {
  return {
    ax: (vel.vx - prevVel.vx) / dt,
    ay: (vel.vy - prevVel.vy) / dt,
    magnitude: Math.sqrt(
      Math.pow((vel.vx - prevVel.vx) / dt, 2) +
      Math.pow((vel.vy - prevVel.vy) / dt, 2)
    )
  }
}

export function emotionalDirection(vx, vy) {
  return Math.atan2(vy, vx)
}
