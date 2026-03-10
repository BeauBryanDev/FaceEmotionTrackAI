import EmotionalPhaseSpace from "./EmotionalPhaseSpace"
import EmotionVectorRadar from "./EmotionVectorRadar"
import EmotionFlowField from "./EmotionFlowField"

import {
  emotionalEntropy,
  emotionalStability,
  detectEmotionRegime
} from "../../core/affective/emotionMetrics"


export default function EmotionalIntelligencePanel({ data = [], current = { valence: 0, arousal: 0 } }) {
  const volatility = data.length
    ? data.reduce((sum, p) => sum + Math.abs(Number(p.arousal) || 0), 0) / data.length
    : 0

  // Fix: map to dominant_emotion and provide fallback to current.entropy for real-time precision
  const historicalEntropy = emotionalEntropy(data.map(d => d.dominant_emotion).filter(Boolean))
  const entropy = current.entropy !== undefined && current.entropy !== null ? current.entropy : historicalEntropy

  const stability = emotionalStability(volatility)

  const regime = detectEmotionRegime({
    valence: Number(current.valence) || 0,
    arousal: Number(current.arousal) || 0,
    volatility
  })

  return (

    <div className="grid grid-cols-2 gap-6">

      <EmotionalPhaseSpace data={data} />

      <EmotionVectorRadar
        valence={current.valence}
        arousal={current.arousal}
      />

      <EmotionFlowField />

      <div className="bg-black/40 border border-purple-700 rounded-xl p-4">

        <h2 className="text-purple-400 font-mono mb-3">
          Emotional Regime
        </h2>

        <div className="space-y-2 text-sm font-mono">

          <div>
            Entropy
            <span className="text-purple-300 ml-2">
              {entropy.toFixed(3)}
            </span>
          </div>

          <div>
            Stability
            <span className="text-purple-300 ml-2">
              {stability.toFixed(3)}
            </span>
          </div>

          <div>
            Volatility
            <span className="text-purple-300 ml-2">
              {volatility.toFixed(3)}
            </span>
          </div>

          <div className="pt-2 border-t border-purple-800">

            Regime

            <span className="ml-2 text-purple-400 uppercase">
              {regime}
            </span>

          </div>

        </div>

      </div>


    </div>

  )
}
