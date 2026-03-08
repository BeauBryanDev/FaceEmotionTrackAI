import { useMemo } from "react"

export default function EmotionVectorRadar({ valence, arousal }) {

  const angle = useMemo(() => {
    return Math.atan2(arousal, valence)
  }, [valence, arousal])

  const magnitude = Math.sqrt(valence*valence + arousal*arousal)

  const x = 150 + magnitude * 100 * Math.cos(angle)
  const y = 150 - magnitude * 100 * Math.sin(angle)

  return (
    <div className="bg-black/40 border border-purple-700 rounded-xl p-4">

      <svg width="300" height="300">

        <circle
          cx="150"
          cy="150"
          r="120"
          stroke="#9333ea"
          strokeWidth="2"
          fill="none"
        />

        <line
          x1="150"
          y1="150"
          x2={x}
          y2={y}
          stroke="#c084fc"
          strokeWidth="3"
        />

      </svg>

    </div>
  )
}
