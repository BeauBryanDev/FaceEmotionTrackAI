import React, { useMemo } from "react"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from "recharts"

/**
 * Custom HUD tooltip
 */
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null

  return (
    <div className="bg-purple-950/95 border-l-2 border-neon-purple p-3 backdrop-blur-md shadow-[0_0_20px_rgba(170,0,255,0.3)]">
      <p className="font-mono text-[10px] text-purple-400 mb-2">
        {label}
      </p>

      {payload.map((p) => (
        <div key={p.dataKey} className="flex justify-between gap-4 font-mono text-[10px]">
          <span style={{ color: p.color }}>
            {p.dataKey.toUpperCase()}
          </span>

          <span className="text-neon-purple font-bold">
            {(p.value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  )
}

const EmotionTimeline = ({ data }) => {
  const getEmotionValue = (row, key) => {
    if (!row || typeof row !== "object") return 0
    if (typeof row[key] === "number") return row[key]
    if (typeof row[key.toLowerCase()] === "number") return row[key.toLowerCase()]
    if (row.emotion_scores && typeof row.emotion_scores === "object") {
      const nested = row.emotion_scores
      if (typeof nested[key] === "number") return nested[key]
      if (typeof nested[key.toLowerCase()] === "number") return nested[key.toLowerCase()]
    }
    return 0
  }

  /**
   * Format timestamps for display
   */
  const formatted = useMemo(() => {
    const source = Array.isArray(data) ? data : Array.isArray(data?.timeline) ? data.timeline : []

    // Slice to last 30 records to keep the graph focused and performant
    const recentSource = source.slice(-30);

    return recentSource.map((d) => ({
      ...d,
      Anger: Number(getEmotionValue(d, "Anger")),
      Contempt: Number(getEmotionValue(d, "Contempt")),
      Disgust: Number(getEmotionValue(d, "Disgust")),
      Fear: Number(getEmotionValue(d, "Fear")),
      Happiness: Number(getEmotionValue(d, "Happiness")),
      Neutral: Number(getEmotionValue(d, "Neutral")),
      Sadness: Number(getEmotionValue(d, "Sadness")),
      Surprise: Number(getEmotionValue(d, "Surprise")),
      time: new Date(d.timestamp).toLocaleTimeString()
    }))
  }, [data])

  /**
   * Emotion color palette (cyberpunk neon)
   */
  const colors = {
    Anger: "#ff0055",
    Contempt: "#ff8800",
    Disgust: "#00ffaa",
    Fear: "#60057cff",
    Happiness: "#bf00ff",
    Neutral: "#aaaaaa",
    Sadness: "#ffaa00",
    Surprise: "#ff33ff"
  }

  const emotionColors = {
    Anger: "#ff0044",
    Contempt: "#ff8800",
    Disgust: "#00ffaa",
    Fear: "#60057cff",
    Happiness: "#bf00ff",
    Neutral: "#aaaaaa",
    Sadness: "#ffaa00",
    Surprise: "#ff33ff"
  }

  const processed = useMemo(() => {
    if (!formatted?.length) return []

    const emotions = [
      "Anger",
      "Contempt",
      "Disgust",
      "Fear",
      "Happiness",
      "Neutral",
      "Sadness",
      "Surprise"
    ]

    return formatted.map((d) => {
      let dominant = "Neutral"
      let max = -1

      emotions.forEach((e) => {
        if (Number(d[e]) > max) {
          max = Number(d[e])
          dominant = e
        }
      })

      return {
        ...d,
        dominant
      }
    })
  }, [formatted])

  return (
    <div className="w-full h-full min-h-[320px] bg-surface-1/40 border border-purple-800/40 p-4 relative overflow-hidden">

      {/* header */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-[10px] text-purple-400 font-mono tracking-widest">
          07 // EMOTION_TIMELINE_STREAM
        </span>

        <div className="flex items-center gap-3">
          <span className="text-[9px] text-purple-600 font-mono">
            RECORDS: {formatted.length}
          </span>
          <div className="flex gap-1">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="w-1 h-1 bg-neon-purple animate-pulse"
                style={{ animationDelay: `${i * 0.2}s` }}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={formatted}>

            <CartesianGrid
              stroke="#6b21a8"
              strokeOpacity={0.2}
              strokeDasharray="3 3"
            />

            <XAxis
              dataKey="time"
              tick={{
                fill: "#9f7aea",
                fontSize: 9,
                fontFamily: "Share Tech Mono"
              }}
              axisLine={{ stroke: "#6b21a8" }}
              tickLine={false}
            />

            <YAxis
              domain={[0, 1]}
              tick={{
                fill: "#9f7aea",
                fontSize: 9,
                fontFamily: "Share Tech Mono"
              }}
              axisLine={{ stroke: "#6b21a8" }}
              tickLine={false}
            />

            <Tooltip content={<CustomTooltip />} />

            <Legend
              wrapperStyle={{
                fontSize: "10px",
                fontFamily: "Share Tech Mono",
                color: "#9f7aea"
              }}
            />

            {Object.keys(colors).map((emotion) => (
              <Line
                key={emotion}
                type="monotone"
                dataKey={emotion}
                stroke={colors[emotion]}
                strokeWidth={2}
                dot={formatted.length <= 1}
                connectNulls
                isAnimationActive
                animationDuration={500}
              />
            ))}

          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex justify-between mt-2 text-[9px] font-mono text-purple-500">
        <span>DOMINANT_SIGNAL_TRACK</span>
        <span>AI_EMOTION_CLASSIFIER</span>
      </div>

      <div className="mt-3 flex h-4 w-full overflow-hidden border border-purple-900">
        {processed.map((p, i) => (
          <div
            key={i}
            className="flex-1 transition-all"
            style={{
              background: emotionColors[p.dominant],
              opacity: 0.8
            }}
            title={p.dominant}
          />
        ))}
      </div>

      {formatted.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="font-mono text-[11px] text-purple-600">No timeline records available</span>
        </div>
      )}

      {/* background cyber grid */}
      <div className="absolute inset-0 pointer-events-none opacity-10">
        <div className="absolute inset-0 bg-cyber-grid bg-[length:20px_20px]" />
      </div>

    </div>
  )
}

export default EmotionTimeline
