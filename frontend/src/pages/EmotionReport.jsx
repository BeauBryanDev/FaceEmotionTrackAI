import { useEffect, useState, useCallback } from "react"
import {
  getEmotionSummary,
  getEmotionScores,
  getEmotionHistory,
  getEmotionDetails,
  getEmotionScoresChart,
  getEmotionTimeline
} from "../api/emotions"

import EmotionRadar from "../components/EmotionRadar"
import { Activity, ChevronLeft, ChevronRight, RefreshCw } from "lucide-react"
import EmotionIntelligencePanel from "../components/EmotionIntelligencePanel"


/* ------------------------------------------------ */
/* Emotion Color Map                                */
/* ------------------------------------------------ */

const EMOTION_COLORS = {
  Happiness: "#00ff88",
  Neutral: "#cc44ff",
  Anger: "#ff4466",
  Sadness: "#4488ff",
  Fear: "#ff8800",
  Disgust: "#44ffcc",
  Surprise: "#ffff00",
  Contempt: "#ff44aa",
}

/* ------------------------------------------------ */
/* Card Container                                   */
/* ------------------------------------------------ */

const Card = ({ children }) => (
  <div className="bg-purple-950/80 border border-purple-800 relative overflow-hidden">
    <div className="absolute top-0 left-0 w-3 h-3 border-t border-l border-purple-500"/>
    <div className="absolute bottom-0 right-0 w-3 h-3 border-b border-r border-purple-500"/>
    <div className="p-4">{children}</div>
  </div>
)

/* ------------------------------------------------ */
/* Emotion Bar                                      */
/* ------------------------------------------------ */

const EmotionBar = ({ emotion, percentage }) => {

  const color = EMOTION_COLORS[emotion] || "#cc44ff"

  return (
    <div className="flex items-center gap-3 text-xs font-mono">
      <div className="w-24 text-purple-300">
        {emotion.toUpperCase()}
      </div>

      <div className="flex-1 h-1.5 bg-purple-900 border border-purple-800">
        <div
          className="h-full"
          style={{
            width: `${percentage}%`,
            background: color,
            boxShadow: `0 0 8px ${color}`
          }}
        />
      </div>

      <div className="w-12 text-right text-purple-400">
        {percentage.toFixed(1)}%
      </div>
    </div>
  )
}

/* ------------------------------------------------ */
/* Detection Row                                    */
/* ------------------------------------------------ */

const TableRow = ({ record }) => {

  const color = EMOTION_COLORS[record.dominant_emotion] || "#cc44ff"

  const ts = new Date(record.timestamp)
  const formatted = ts.toISOString().replace("T", " ").slice(0, 19)

  return (
    <tr className="border-b border-purple-900 text-sm font-mono">
      <td className="p-2 text-purple-500">{record.id}</td>
      <td className="p-2">{formatted}</td>
      <td className="p-2">
        <span
          style={{
            color,
            border: `1px solid ${color}60`,
            padding: "2px 6px",
            background: `${color}20`
          }}
        >
          {record.dominant_emotion}
        </span>
      </td>
      <td className="p-2 text-purple-200">
        {(record.confidence * 100).toFixed(1)}%
      </td>
    </tr>
  )
}

/* ------------------------------------------------ */
/* Main Page                                        */
/* ------------------------------------------------ */

const EmotionReport = () => {

  const [summary,setSummary] = useState(null)
  const [scores,setScores] = useState(null)
  const [history,setHistory] = useState({records:[],total_pages:1})
  const [details,setDetails] = useState(null)
  const [chart,setChart] = useState(null)
  const [timeline, setTimeline] = useState([])

  const [page,setPage] = useState(1)
  const [selectedEmotion,setSelectedEmotion] = useState("Happiness")

  const [loading,setLoading] = useState(true)

  const PAGE_SIZE = 12

  const loadAll = useCallback(async (currentPage=1) => {

    setLoading(true)

    const [
      summaryData,
      scoreData,
      historyData,
      detailsData,
      chartData,
      timelineData
    ] = await Promise.all([
      getEmotionSummary(),
      getEmotionScores(1),
      getEmotionHistory({page:currentPage,page_size:PAGE_SIZE}),
      getEmotionDetails(selectedEmotion),
      getEmotionScoresChart(),
      getEmotionTimeline(200)
    ])

    setSummary(summaryData)
    setScores(scoreData?.records?.[0]?.emotion_scores || null)
    setHistory(historyData)
    setDetails(detailsData)
    setChart(chartData)
    setTimeline(timelineData?.timeline || [])

    setLoading(false)

  },[selectedEmotion])

  useEffect(()=>{
    loadAll(page)
  },[page,selectedEmotion,loadAll])

  if(loading)
  return (
    <div className="p-8 font-mono text-purple-400">
      LOADING EMOTION REPORT...
    </div>
  )

  return (

  <div className="p-4 sm:p-6 flex flex-col gap-6">

  {/* Header */}

  <div className="flex flex-col md:flex-row md:justify-between md:items-end border-b border-purple-800 pb-4 gap-3">

    <div>
      <div className="text-xs font-mono text-purple-500">
        SYSTEM / EMOTION ANALYSIS
      </div>

      <h1 className="text-xl sm:text-2xl text-purple-200 font-bold flex items-center gap-2">
        <Activity size={18}/>
        EMOTION REPORT
      </h1>
    </div>

    <div className="font-mono text-xs sm:text-sm text-purple-400">
      TOTAL DETECTIONS: {summary?.total_detections}
    </div>

  </div>

  {/* Row 1 */}

  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

    <Card>

      <div className="text-xs text-purple-400 mb-3">
        DOMINANT EMOTION
      </div>

      <div className="text-3xl sm:text-4xl text-neon-purple font-bold">
        {summary?.dominant_emotion}
      </div>

      <div className="mt-4 flex flex-col gap-2">
        {summary?.emotion_stats?.map(stat=>(
          <EmotionBar
            key={stat.emotion}
            emotion={stat.emotion}
            percentage={stat.percentage}
          />
        ))}
      </div>

    </Card>

    <Card>

      <div className="text-xs text-purple-400 mb-3">
        RADAR DISTRIBUTION
      </div>

      <div className="h-64">
        <EmotionRadar emotionScores={scores}/>
      </div>

    </Card>

  </div>

  {/* Row 2 */}

  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

    <Card>

      <div className="flex justify-between mb-3">

        <div className="text-xs text-purple-400">
          EMOTION DETAILS
        </div>

        <select
          value={selectedEmotion}
          onChange={(e)=>setSelectedEmotion(e.target.value)}
          className="bg-purple-950 border border-purple-700 text-xs font-mono"
        >
          {Object.keys(EMOTION_COLORS).map(e=>(
            <option key={e}>{e}</option>
          ))}
        </select>

      </div>

      {details?.emotion_stats?.map(d=>(
        <div key={d.class} className="flex justify-between text-sm font-mono">
          <span>{d.class}</span>
          <span>{(d.confidence*100).toFixed(1)}%</span>
        </div>
      ))}

    </Card>

    <Card>

      <div className="text-xs text-purple-400 mb-3">
        AGGREGATE SCORES
      </div>

      {chart?.emotion_stats?.map(c=>(
        <EmotionBar
          key={c.class}
          emotion={c.class}
          percentage={c.percentage}
        />
      ))}

    </Card>

  </div>
  {/* Emotion Intelligence Panel */}
  <EmotionIntelligencePanel timeline={timeline}/>


  {/* Row 3 — Detection History */}

  <Card>

  <div className="flex justify-between mb-4">

    <div className="text-xs text-purple-400">
      DETECTION HISTORY
    </div>

    <div className="text-xs text-purple-500">
      PAGE {page} / {history.total_pages}
    </div>

  </div>

  <div className="overflow-x-auto">
  <table className="w-full text-left min-w-[640px]">
  <thead>
  <tr className="text-xs font-mono text-purple-400 border-b border-purple-800">
  <th className="p-2">ID</th>
  <th className="p-2">TIMESTAMP</th>
  <th className="p-2">EMOTION</th>
  <th className="p-2">CONFIDENCE</th>
  </tr>
  </thead>

  <tbody>
  {history.records.map(r=>(
    <TableRow key={r.id} record={r}/>
  ))}
  </tbody>

  </table>
  </div>

  <div className="flex justify-end gap-2 mt-4">

  <button
  onClick={()=>setPage(p=>Math.max(1,p-1))}
  className="border border-purple-700 px-2 py-1"
  >
  <ChevronLeft size={14}/>
  </button>

  <button
  onClick={()=>setPage(p=>Math.min(history.total_pages,p+1))}
  className="border border-purple-700 px-2 py-1"
  >
  <ChevronRight size={14}/>
  </button>

  </div>

  </Card>

  </div>
  )
}

export default EmotionReport
