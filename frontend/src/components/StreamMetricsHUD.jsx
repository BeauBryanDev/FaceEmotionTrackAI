import React from "react"

export default function StreamMetricsHUD({ data }) {
  const emotion = data?.emotion?.dominant_emotion ?? "--"
  const ear = data?.geometry?.ear?.ear
  const mar = data?.geometry?.mar?.mar
  const pose = data?.geometry?.head_pose?.pose_label ?? "--"
  const smile = data?.geometry?.expressions?.smile_score
  const talk = data?.geometry?.expressions?.talk_score
  const happy = data?.geometry?.expressions?.happy_score
  const engagement = data?.geometry?.expressions?.engagement_score

  const fmt = (value) => (typeof value === "number" ? value.toFixed(2) : "--")
  const valueClass = (level) => {
    if (level === "good") return "text-green-400"
    if (level === "warn") return "text-neon-purple"
    return "text-red-500"
  }
  const numericLevel = (value, goodMin, warnMin = null, inverse = false) => {
    if (typeof value !== "number") return "warn"
    if (!inverse) {
      if (value >= goodMin) return "good"
      if (warnMin !== null && value >= warnMin) return "warn"
      return "critical"
    }
    if (value <= goodMin) return "good"
    if (warnMin !== null && value <= warnMin) return "warn"
    return "critical"
  }
  const poseLevel = () => {
    const p = String(pose || "").toUpperCase()
    if (["FORWARD", "CENTER", "CENTRAL", "FRONTAL"].some((k) => p.includes(k))) return "good"
    if (["LEFT", "RIGHT", "UP"].some((k) => p.includes(k))) return "warn"
    return "critical"
  }

  return (
    <div className="bg-surface-1 border border-purple-800 p-4 font-mono text-xs">
      <div className="text-purple-400 mb-2 tracking-widest">AFFECTIVE SIGNAL HUD</div>

      <div className="flex justify-between mb-1">
        <span className="text-purple-300">Dominant Emotion</span>
        <span className="text-neon-purple font-bold">{emotion}</span>
      </div>
      <div className="flex justify-between mb-1">
        <span className="text-purple-300">EAR</span>
        <span className={valueClass(numericLevel(ear, 0.23, 0.16))}>{fmt(ear)}</span>
      </div>
      <div className="flex justify-between mb-1">
        <span className="text-purple-300">MAR</span>
        <span className={valueClass(numericLevel(mar, 0.45, 0.65, true))}>{fmt(mar)}</span>
      </div>
      <div className="flex justify-between mb-1">
        <span className="text-purple-300">Head Pose</span>
        <span className={valueClass(poseLevel())}>{pose}</span>
      </div>
      <div className="flex justify-between mb-1">
        <span className="text-purple-300">Smile Score</span>
        <span className={valueClass(numericLevel(smile, 0.5, 0.25))}>{fmt(smile)}</span>
      </div>
      <div className="flex justify-between mb-1">
        <span className="text-purple-300">Talking Activity</span>
        <span className={valueClass(numericLevel(talk, 0.35, 0.65, true))}>{fmt(talk)}</span>
      </div>
      <div className="flex justify-between mb-1">
        <span className="text-purple-300">Happy Score</span>
        <span className={valueClass(numericLevel(happy, 0.55, 0.25))}>{fmt(happy)}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-purple-300">Engagement Score</span>
        <span className={valueClass(numericLevel(engagement, 0.6, 0.35))}>{fmt(engagement)}</span>
      </div>
    </div>
  )
}
