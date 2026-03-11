import React, { useEffect, useState } from 'react'
import { getEmotionSummary, getEmotionScores, getEmotionHistory, getEmotionDetails, getEmotionScoresChart, getEmotionTimeline } from '../api/emotions'
import Text from '../components/ui/Text'
import EmotionRadar from '../components/EmotionRadar'
import EmotionDistributionPie from '../components/EmotionDistributionPie'
import SentimentDoughnut from '../components/SentimentDoughnut'
import EmotionHistoryHistogram from '../components/EmotionHistoryHistogram'
import ConfidenceRadar from '../components/ui/ConfidentRadar'
import EmotionTimeline from '../components/EmotionTimeline'
import NeuralStabilityMeter from '../components/NeuralStabilityMeter'



const Emotions = () => {
  const [summary, setSummary] = useState(null)
  const [scores, setScores] = useState(null)
  const [timeline, setTimeline] = useState([])
  const [recent, setRecent] = useState([])
  const [details, setDetails] = useState(null)
  const [selectedEmotion, setSelectedEmotion] = useState('Happiness')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [entropy, setEntropy] = useState(null)


  const formatTs = (iso) => {
    if (!iso) return ''
    const d = new Date(iso)

    return d.toISOString().replace('T', ' ').substring(0, 19)
  }
  const entropyForRadar = entropy ?? recent?.[0]?.entropy ?? null

  useEffect(() => {
    const run = async () => {
      try {
        setLoading(true)
        setError(null)
        const [summaryData, scoreData, historyData, _chartData, detailsData, timelineData] = await Promise.all([
          getEmotionSummary(),
          getEmotionScores(1),
          getEmotionHistory({ page: 1, page_size: 7 }),
          getEmotionScoresChart(),
          getEmotionDetails(selectedEmotion),
          getEmotionTimeline(200),
        ])
        setSummary(summaryData)
        setScores(scoreData?.records?.[0]?.emotion_scores || null)
        setEntropy(scoreData?.records?.[0]?.entropy ?? null)
        setRecent(historyData?.records || [])
        setDetails(detailsData)
        setTimeline(timelineData?.timeline || [])
      } catch (e) {
        setError('FAILED TO LOAD EMOTION DATA')
      } finally {
        setLoading(false)
      }
    }
    run()
  }, [selectedEmotion])


  if (loading) {
    return (
      <div className="p-6 h-screen flex items-center justify-center bg-surface-0">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-2 border-neon-purple border-t-transparent rounded-full animate-spin shadow-neon-sm" />
          <Text variant="mono" className="animate-pulse text-purple-400 tracking-[0.3em]">
            INITIALIZING_NEURAL_MONITOR...
          </Text>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 h-screen flex items-center justify-center bg-surface-0">
        <div className="bg-purple-900/20 border border-red-500/50 p-8 flex flex-col items-center gap-4">
          <div className="text-red-500 text-4xl">⚠</div>
          <Text variant="mono" className="text-red-500 font-bold">{error}</Text>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-red-500/10 border border-red-500 text-red-500 font-mono text-xs hover:bg-red-500 hover:text-white transition-all"
          >
            RETRY_SYSTEM_LINK
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 sm:p-6 min-h-[calc(100vh-80px)] bg-surface-0 bg-cyber-grid flex flex-col gap-6 relative overflow-hidden">

      {/* HEADER DECOR */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-neon-purple/5 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-64 h-64 bg-neon-violet/5 blur-[120px] pointer-events-none" />

      {/* DASHBOARD TOP BAR */}
      <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-purple-800/50 pb-6 gap-4 relative">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-3">
            <div className="w-1 h-6 bg-neon-purple shadow-neon-sm" />
            <Text variant="h2" className="tracking-tighter text-2xl sm:text-3xl">NEURAL_EMOTION_HUD</Text>
          </div>
          <Text variant="mono" className="text-purple-500 text-[10px] tracking-[0.2em] ml-4">SYSTEM_STATUS: ACTIVE // DATA_FLOW: STABLE</Text>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-6">
          <div className="flex flex-col sm:items-end">
            <span className="text-[10px] font-mono text-purple-500 uppercase tracking-widest">Aggregate Detections</span>
            <span className="text-xl sm:text-2xl font-display text-neon-purple glow-sm">{summary?.total_detections || 0}</span>
          </div>
          <div className="hidden sm:block h-10 w-[1px] bg-purple-800/50" />
          <div className="flex flex-col sm:items-end">
            <span className="text-[10px] font-mono text-purple-500 uppercase tracking-widest">Dominant State</span>
            <span className="text-xl sm:text-2xl font-display text-neon-purple glow-sm">{(summary?.dominant_emotion || 'N/A').toUpperCase()}</span>
          </div>
        </div>
      </div>

      {/* MAIN COMMAND GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">

        {/* COLUMN 1: RADAR & SENTIMENT */}
        <div className="flex flex-col gap-6">
          {/* RADAR CORE */}
          <div className="bg-surface-1/40 border border-purple-800/40 p-1 relative group">
            <div className="absolute -top-[1px] -left-[1px] w-6 h-6 border-t border-l border-neon-purple z-10" />
            <div className="p-4 flex flex-col h-[320px] sm:h-[400px]">
              <div className="flex items-center justify-between mb-4">
                <Text variant="subtext" className="text-[10px] text-purple-400">01 // NEURAL_RADAR</Text>
                <div className="flex gap-1">
                  {[1, 2, 3].map(i => <div key={i} className="w-1 h-1 bg-neon-purple animate-pulse" style={{ animationDelay: `${i * 0.2}s` }} />)}
                </div>
              </div>
              <div className="flex-1">
                <EmotionRadar emotionScores={scores} />
              </div>
            </div>
          </div>

          {/* SENTIMENT DOUGHNUT */}
          <div className="bg-surface-1/40 border border-purple-800/40 p-1 relative overflow-hidden">
            <div className="p-4 flex flex-col h-[260px] sm:h-[320px]">
              <Text variant="subtext" className="text-[10px] text-purple-400 mb-2">02 // SENTIMENT_INDEX</Text>
              <div className="flex-1">
                <SentimentDoughnut stats={summary?.emotion_stats} />
              </div>
            </div>
          </div>

          {/* CONFIDENCE RADAR */}
          <div className="bg-surface-1/40 border border-purple-800/40 p-1 relative">
            <div className="p-4 flex flex-col h-[240px] sm:h-[320px]">
              <Text variant="subtext" className="text-[10px] text-purple-400 mb-2">
                02 // MODEL_CONFIDENCE
              </Text>

              <div className="flex-1">
                <ConfidenceRadar entropy={entropyForRadar} />
              </div>
            </div>
         </div>
        </div>

        {/* COLUMN 2: DISTRIBUTION & HISTOGRAM */}
        <div className="flex flex-col gap-6">
          {/* PIE DISTRIBUTION */}
          <div className="bg-surface-1/40 border border-purple-800/40 p-1 relative">
            <div className="p-4 flex flex-col h-[260px] sm:h-[300px]">
              <Text variant="subtext" className="text-[10px] text-purple-400 mb-2">03 // CLASS_DISTRIBUTION</Text>
              <div className="flex-1">
                <EmotionDistributionPie data={summary?.emotion_stats} />
              </div>
            </div>
          </div>

          {/* HISTOGRAM */}
          <div className="bg-surface-1/40 border border-purple-800/40 p-1 relative">
            <div className="p-4 flex flex-col h-[260px] sm:h-[320px]">
              <div className="flex items-center justify-between mb-4">
                <Text variant="subtext" className="text-[10px] text-purple-400">04 // INTENSITY_HISTOGRAM</Text>
                <select
                  value={selectedEmotion}
                  onChange={(e) => setSelectedEmotion(e.target.value)}
                  className="bg-purple-950 border border-purple-700 text-purple-400 font-mono text-[9px] px-2 py-0.5 outline-none focus:border-neon-purple uppercase"
                >
                  {['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'].map((emo) => (
                    <option key={emo} value={emo}>{emo}</option>
                  ))}
                </select>
              </div>
              <div className="flex-1">
                <EmotionHistoryHistogram data={details?.emotion_stats} />
              </div>
            </div>
          </div>
          {/* NEURAL STABILITY METER */}
          <NeuralStabilityMeter entropy={entropy} />
          <div className="flex justify-between mt-2 text-[9px] font-mono text-purple-500">
          <span>DOMINANT_SIGNAL_TRACK</span>
          <span>AI_EMOTION_CLASSIFIER</span>
        </div>

        </div>

        {/* COLUMN 3: RECENT LOG & READOUTS */}
        <div className="flex flex-col gap-6">
          {/* STATS BARS */}
          <div className="bg-surface-1/40 border border-purple-800/40 p-4 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-2">
              <div className="w-16 h-[1px] bg-neon-purple/20 rotate-45 transform translate-x-4 -translate-y-4" />
            </div>
            <Text variant="subtext" className="text-[10px] text-purple-400 mb-4 uppercase">05 // TELEMETRY_FEED</Text>
            <div className="space-y-4">
              {(summary?.emotion_stats || []).map((stat) => (
                <div key={stat.emotion} className="group cursor-help">
                  <div className="flex justify-between mb-1">
                    <span className="font-mono text-[9px] text-purple-300 group-hover:text-neon-purple transition-colors uppercase">{stat.emotion}</span>
                    <span className="font-mono text-[9px] text-purple-500">{stat.percentage.toFixed(1)}%</span>
                  </div>
                  <div className="h-1 bg-purple-900 overflow-hidden relative">
                    <div
                      className="h-full bg-neon-purple shadow-neon-sm transition-all duration-1000"
                      style={{ width: `${stat.percentage}%` }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent w-full h-full -translate-x-full group-hover:animate-scan-fast" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* RECENT DETECTIONS TABLE */}
          <div className="bg-surface-1/40 border border-purple-800/40 p-4 relative flex-1 overflow-hidden flex flex-col">
            <Text variant="subtext" className="text-[10px] text-purple-400 mb-4 uppercase">06 // EVENT_LOG</Text>
            <div className="flex-1 overflow-y-auto scrollbar-cyber pr-2">
              <div className="space-y-2">
                {recent.slice(0, 7).map((r) => (
                  <div key={r.id} className="flex flex-col gap-1 p-2 bg-purple-950/40 border-l border-purple-800 hover:border-neon-purple transition-all group">
                    <div className="flex justify-between items-center px-1">
                      <span className="font-mono text-[8px] text-purple-600">ID: {String(r.id).padStart(4, '0')}</span>
                      <span className="font-mono text-[8px] text-purple-600">{formatTs(r.timestamp)}</span>
                    </div>
                    <div className="flex justify-between items-center px-2">
                      <span className="font-mono text-[11px] text-purple-200 uppercase group-hover:text-neon-purple transition-colors">{r.dominant_emotion}</span>
                      <span className="font-mono text-[11px] text-neon-purple font-black">{(r.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="px-2">
                      <span className="font-mono text-[9px] text-purple-500">
                        ENTROPY: {r.entropy !== null && r.entropy !== undefined ? Number(r.entropy).toFixed(3) : '--'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

      </div>
      {/* TIMELINE */}
          <div className="mt-6">
            <EmotionTimeline data={timeline} />
          </div>

      {/* FOOTER SYSTEM METRICS */}
      <div className="mt-4 flex items-center justify-between text-[8px] font-mono text-purple-600 tracking-widest border-t border-purple-800/30 pt-4">
        <div className="flex gap-4">
          <span>COORDS: [34.0522, -118.2437]</span>
          <span>UI_VER: 4.2.0-STABLE</span>
          <span>NEURAL_ENGINE: V3-PRO</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-neon-purple shadow-neon-sm animate-pulse" />
          <span>DATALINK: SECURE_ENCRYPTED</span>
        </div>
      </div>

    </div>
  )
}

export default Emotions
