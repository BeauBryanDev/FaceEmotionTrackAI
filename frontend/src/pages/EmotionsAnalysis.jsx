import { useEffect, useState } from "react"
import RussellCircumplexChart from "../components/RussellCircumplexChart"
import EmotionalTrajectory from "../components/EmotionalTrajectory"
import EmotionIntensityMeter from "../components/EmotionIntensityMeter"
import RussellQuadrants from "../components/RussellQuadrants"
import EmotionMomentum from "../components/EmotionMomentum"
import EmotionDrift from "../components/EmotionDrift"
import EmotionTurbulence from "../components/EmotionTurbulence"
import EmotionPhaseSpace from "../components/EmotionPhaseSpace"
import EmotionVectorField from "../components/EmotionVectorField"
import EmotionTimeline from "../components/EmotionTimeline"

import { calculateRussellCoordinates, getQuadrant } from "../utils/russellMapping"
import { computeEmotionDynamics } from "../utils/emotionDynamics"

import { getEmotionScores } from "../api/emotions"

import EmotionalIntelligencePanel from "../components/ui/EmotionalIntelligencePanel"

/*
Emotion Analysis (Russell Model)

├─ Russell Circumplex Plane (scatter + trajectory)
├─ Emotional Trajectory (time series X,Y)
├─ Emotional Intensity (distance from origin)
├─ Quadrant Distribution
└─ Current Emotional Vector
*/

const EmotionsAnalysis = () => {

  const [points, setPoints] = useState([])

  useEffect(() => {

    async function load() {
      const data = await getEmotionScores()

      const coords = data.records.map((r, i) => {

        const p = Object.values(r.emotion_scores)

        const { valence, arousal } = calculateRussellCoordinates(p)

        return {
          valence,
          arousal,
          entropy: r.entropy,
          dominant_emotion: r.dominant_emotion,
          time: i
        }
      })

      setPoints(coords)
    }

    load()

  }, [])

  const current = points[points.length - 1] || { valence: 0, arousal: 0 }

  const dynamics = computeEmotionDynamics(points)
  const tacticalState = getQuadrant(current.valence, current.arousal)

  return (
    <div className="p-4 sm:p-6 space-y-6 bg-surface-0 bg-cyber-grid min-h-[calc(100vh-80px)]">

      <div className="flex flex-col md:flex-row md:justify-between md:items-end border-b border-purple-800 pb-4 gap-3">
        <div>
          <h1 className="text-lg sm:text-xl text-purple-200 font-bold tracking-[0.2em] flex items-center gap-2">
            NEURAL_AFFECT_CORE // V4.0
          </h1>
          <p className="font-mono text-[9px] text-purple-500 mt-1 uppercase tracking-widest">
            Tactical Emotional Analysis (Circumplex Model)
          </p>
        </div>
        <div className="bg-purple-950 border border-neon-purple/50 px-3 py-1 text-center w-full md:w-auto">
          <span className="block text-[7px] text-purple-400 font-mono tracking-tighter uppercase">Current Tactical State</span>
          <span className="text-neon-purple font-mono font-black text-sm glow-sm uppercase">
            {tacticalState.label}
          </span>
        </div>
      </div>

      <RussellCircumplexChart data={points} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        <EmotionIntensityMeter
          valence={current.valence}
          arousal={current.arousal}
        />

        <RussellQuadrants data={points} />

      </div>

      <EmotionalTrajectory data={points} />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

        <EmotionMomentum dynamics={dynamics} />
        <EmotionDrift dynamics={dynamics} />
        <EmotionTurbulence data={points} />

      </div>

      <EmotionPhaseSpace dynamics={dynamics} />

      <EmotionVectorField dynamics={dynamics} />

      <EmotionalIntelligencePanel data={points} current={current} />

    </div>

  )
}

export default EmotionsAnalysis
