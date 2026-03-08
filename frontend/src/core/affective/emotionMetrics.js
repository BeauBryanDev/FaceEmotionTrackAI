export function emotionalEnergy(valence, arousal) {

  return 0.5 * (valence * valence + arousal * arousal)
}

export function emotionalMomentum(energy, velocityMagnitude) {
  return energy * velocityMagnitude
}

export function emotionalDrift(current, baseline) {
  const dx = current.valence - baseline.valence
  const dy = current.arousal - baseline.arousal

  return Math.sqrt(dx * dx + dy * dy)
}

export function emotionalTurbulence(points) {
  const mean =
    points.reduce((sum, p) => sum + Math.sqrt(p.valence ** 2 + p.arousal ** 2), 0) /
    points.length

  const variance =
    points.reduce((sum, p) => {
      const mag = Math.sqrt(p.valence ** 2 + p.arousal ** 2)
      return sum + Math.pow(mag - mean, 2)
    }, 0) / points.length

  return variance
}

export function emotionalAngularVelocity(prev, current, dt = 1) {
  const theta1 = Math.atan2(prev.arousal, prev.valence)
  const theta2 = Math.atan2(current.arousal, current.valence)

  return (theta2 - theta1) / dt
}

// ---------- Emotional Entropy ----------

export function emotionalEntropy(emotions) {
  const total = emotions.length
  if (!total) return 0

  const counts = {}

  emotions.forEach(e => {
    counts[e] = (counts[e] || 0) + 1
  })

  let entropy = 0

  Object.values(counts).forEach(count => {
    const p = count / total
    entropy -= p * Math.log(p)
  })

  return entropy
}

// ---------- Emotional Stability ----------

export function emotionalStability(volatility) {
  return 1 / (1 + volatility)
}


// ---------- Emotional Regime Detection ----------

export function detectEmotionRegime({ valence, arousal, volatility }) {

  if (volatility > 0.7) return "chaotic"

  if (valence > 0.4 && arousal > 0.4)
    return "excited"

  if (valence < -0.4 && arousal < 0)
    return "depressed"

  if (arousal > 0.5)
    return "agitated"

  return "calm"
}