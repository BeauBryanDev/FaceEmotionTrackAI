import api from './axios'

/**
 * Fetches paginated emotion history with optional filters.
 * @param {object} params - { page, page_size, emotion_filter, date_from, date_to }
 */
export const getEmotionHistory = async (params = {}) => {
  const response = await api.get('/emotions/history', { params })
  return response.data
}

/**
 * Fetches aggregated emotion statistics for the authenticated user.
 * Returns dominant_emotion, total_detections, emotion_stats array.
 */
export const getEmotionSummary = async () => {
  const response = await api.get('/emotions/summary')
  return response.data
}

/**
 * Fetches the last N emotion records that have emotion_scores populated.
 * Used for radar charts and distribution graphs.
 * @param {number} limit - Max records to return (1-50)
 */
export const getEmotionScores = async (limit = 10) => {
  const response = await api.get('/emotions/scores', { params: { limit } })
  return response.data
}

export const getEmotionDetails = async (emotion) => {
  const response = await api.get('/emotions/details', { params: { emotion } })
  return response.data
}

export const getEmotionScoresChart = async () => {
  const response = await api.get('/emotions/scores/chart')
  return response.data
}
/**
 * Saves the current emotion inference result to the database.
 * Called explicitly when the user clicks "SAVE EMOTION" in LiveStream.
 * @param {object} payload - { dominant_emotion, confidence, emotion_scores, user_note }
 */
export const saveEmotion = async (payload) => {
  const response = await api.post('/emotions/save', payload)
  return response.data
}
