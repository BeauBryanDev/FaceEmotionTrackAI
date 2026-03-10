const RUSSELL_EMOTION_VECTORS = {
  Anger: { valence: -0.7, arousal: 0.8 },
  Contempt: { valence: -0.5, arousal: 0.2 },
  Disgust: { valence: -0.8, arousal: 0.3 },
  Fear: { valence: -0.9, arousal: 0.9 },
  Happiness: { valence: 0.9, arousal: 0.7 },
  Neutral: { valence: 0.0, arousal: 0.0 },
  Sadness: { valence: -0.7, arousal: -0.5 },
  Surprise: { valence: 0.2, arousal: 0.9 },
}

const EMOTION_ORDER = [
  'Anger',
  'Contempt',
  'Disgust',
  'Fear',
  'Happiness',
  'Neutral',
  'Sadness',
  'Surprise',
]

/**
 * Tactical Russell Quadrants
 */
export const RUSSELL_QUADRANTS = {
  Q1: { label: 'ALARM_PHASE', color: '#ff4466', description: 'High Arousal / Negative Valence' },
  Q2: { label: 'STRIKE_READY', color: '#00ff88', description: 'High Arousal / Positive Valence' },
  Q3: { label: 'TACTICAL_RECOVERY', color: '#4488ff', description: 'Low Arousal / Positive Valence' },
  Q4: { label: 'DORMANCY_STATE', color: '#ff8800', description: 'Low Arousal / Negative Valence' },
  CENTER: { label: 'NEUTRAL_AXIS', color: '#cccccc', description: 'Equilibrium' },
}

export const getQuadrant = (valence, arousal) => {
  if (Math.abs(valence) < 0.1 && Math.abs(arousal) < 0.1) return RUSSELL_QUADRANTS.CENTER
  if (valence >= 0 && arousal >= 0) return RUSSELL_QUADRANTS.Q2
  if (valence < 0 && arousal >= 0) return RUSSELL_QUADRANTS.Q1
  if (valence < 0 && arousal < 0) return RUSSELL_QUADRANTS.Q4
  return RUSSELL_QUADRANTS.Q3
}

export const calculateRussellCoordinates = (probabilities = []) => {
  if (!Array.isArray(probabilities) || probabilities.length === 0) {
    return { valence: 0, arousal: 0 }
  }

  const values = probabilities.map((p) => Number(p) || 0)
  const sum = values.reduce((acc, curr) => acc + curr, 0)

  if (sum <= 0) {
    return { valence: 0, arousal: 0 }
  }

  const normalized = values.map((p) => p / sum)

  let valence = 0
  let arousal = 0

  EMOTION_ORDER.forEach((emotion, idx) => {
    const weight = normalized[idx] || 0
    const vector = RUSSELL_EMOTION_VECTORS[emotion]
    valence += vector.valence * weight
    arousal += vector.arousal * weight
  })

  return {
    valence: Number(valence.toFixed(4)),
    arousal: Number(arousal.toFixed(4)),
  }
}
