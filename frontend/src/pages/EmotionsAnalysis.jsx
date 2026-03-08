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

import { calculateRussellCoordinates } from "../utils/russellMapping"
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

  const [points,setPoints] = useState([])

  useEffect(()=>{

    async function load()
    {
      const data = await getEmotionScores()

      const coords = data.records.map((r,i)=>{

        const p = Object.values(r.emotion_scores)

        const {valence,arousal} = calculateRussellCoordinates(p)

        return {
          valence,
          arousal,
          time:i
        }
      })

      setPoints(coords)
    }

    load()

  },[])

  const current = points[points.length-1] || {valence:0,arousal:0}

  const dynamics = computeEmotionDynamics(points)


  return (

  <div className="p-6 space-y-6">

    <h1 className="text-xl text-purple-300">
      Emotion Analysis (Russell Model)
    </h1>

    <RussellCircumplexChart data={points}/>

    <div className="grid grid-cols-2 gap-4">

      <EmotionIntensityMeter
        valence={current.valence}
        arousal={current.arousal}
      />

      <RussellQuadrants data={points}/>

    </div>

    <EmotionalTrajectory data={points}/>

    <div className="grid grid-cols-3 gap-4">

        <EmotionMomentum dynamics={dynamics}/>
        <EmotionDrift dynamics={dynamics}/>
        <EmotionTurbulence data={points}/>

    </div>

    <EmotionPhaseSpace dynamics={dynamics}/>

    <EmotionVectorField dynamics={dynamics}/>

    <EmotionalIntelligencePanel data={points} current={current} />

  </div>

  )
}

export default EmotionsAnalysis
