import EntropyTrendChart from "./EntropyTrendChart"
import ModelUncertaintyMeter from "./ModelUncertaintyMeter"
import EmotionVolatility from "./EmotionVolatility"
import PredictionStability from "./PredictionStability"
import EmotionActivityTimeline from "./EmotionActivityTimeline"
import ExpressionSignalsHUD from "./ExpressionSignalsHUD"
import EmotionRadar from "./EmotionRadar"
import EmotionTemporalSignalGraph from "./EmotionTemporalSignalGraph"



const EmotionIntelligencePanel = ({ timeline }) => {

  if (!timeline || timeline.length === 0)
    return null

  return (

    <div className="grid grid-cols-2 gap-4">

      <div className="bg-purple-950 border border-purple-800 p-3">
        <EntropyTrendChart data={timeline} />
      </div>

      <div className="bg-purple-950 border border-purple-800 p-3">
        <EmotionActivityTimeline data={timeline} />
      </div>

      <div className="bg-purple-950 border border-purple-800 p-3">
        <ModelUncertaintyMeter data={timeline} />
      </div>

      <div className="bg-purple-950 border border-purple-800 p-3">
        <EmotionVolatility data={timeline} />
      </div>

      <div className="bg-purple-950 border border-purple-800 p-3 col-span-2">
        <PredictionStability data={timeline} />
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "16px",
        width: "100%"
      }}>

        <div className="bg-purple-950 border border-purple-800 p-3">
          <EmotionRadar emotionScores={timeline[timeline.length - 1]?.emotion_scores} />
        </div>

        <div className="bg-purple-950 border border-purple-800 p-3">
          <ExpressionSignalsHUD data={{
            smile_score: timeline[timeline.length - 1]?.emotion_scores?.Surprise || 0,
            talk_score: 0,
            happy_score: timeline[timeline.length - 1]?.emotion_scores?.Happiness || 0
          }} />
        </div>

      </div>

      <div className="bg-purple-950 border border-purple-800 p-3 col-span-2">
        <EmotionTemporalSignalGraph data={{
          happy_score: timeline[timeline.length - 1]?.emotion_scores?.Happiness || 0
        }} />
      </div>


    </div>

  )
}

export default EmotionIntelligencePanel
