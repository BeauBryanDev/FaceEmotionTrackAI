import React, { useEffect, useMemo, useCallback, useState } from 'react';
import { Activity, ShieldCheck, ShieldAlert, Eye, AlertTriangle, Cpu } from 'lucide-react';
import { useFaceTracking } from '../hooks/useFaceTracking';
import EmotionRadar from './EmotionRadar';
import { saveEmotion } from '../api/emotions'
import { INFERENCE_FRAME } from '../config/inference'
import ConfidentRadar from './ui/ConfidentRadar'


const LiveStream = () => {
  const {
    videoRef,
    results,
    isConnected,
    error,
    startCamera,
    stopCamera,
    fps,
    latency,
    throughput,
    emotionScores,
    events,
  } = useFaceTracking();

  const [saveStatus, setSaveStatus] = useState('IDLE') // IDLE | SAVING | SAVED | ERROR
  const [heatmapEnabled, setHeatmapEnabled] = useState(false)


  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, [startCamera, stopCamera]);

  // Secure JSON structure extraction - memoized to avoid repeated derived work.
  const {
    hasFace,
    bbox,
    emotion,
    liveness,
    eyeState,
    isDrowsy,
    headPose,
    metrics,
    pipeline,
    entropy,
  } = useMemo(() => {
    const faceDetected = results?.status === 'success'
    return {
      hasFace: faceDetected,
      bbox: faceDetected ? results.bbox : null,
      emotion: faceDetected ? results.emotion?.dominant_emotion : 'SCANNING...',
      liveness: faceDetected ? results.liveness : null,
      eyeState: faceDetected ? results.geometry?.ear?.eye_state : null,
      isDrowsy: faceDetected ? results.geometry?.ear?.is_drowsy : false,
      headPose: faceDetected ? results.geometry?.head_pose?.pose_label : null,
      metrics: results?.metrics ?? null,
      pipeline: results?.ml_pipeline ?? null,
      entropy: faceDetected ? (results?.emotion?.entropy ?? null) : null,
    }
  }, [results])
 
  const FRAME_WIDTH = INFERENCE_FRAME.width
  const FRAME_HEIGHT = INFERENCE_FRAME.height


  const bboxStyles = useMemo(() => {
    if (!bbox) return { display: 'none' }
    const [x1, y1, x2, y2] = bbox
    return {
      left: `${(x1 / FRAME_WIDTH) * 100}%`,
      top: `${(y1 / FRAME_HEIGHT) * 100}%`,
      width: `${((x2 - x1) / FRAME_WIDTH) * 100}%`,
      height: `${((y2 - y1) / FRAME_HEIGHT) * 100}%`,
    }
  }, [bbox, FRAME_WIDTH, FRAME_HEIGHT])

  const sortedEmotionEntries = useMemo(
    () => Object.entries(emotionScores).sort((a, b) => b[1] - a[1]),
    [emotionScores]
  )


  const handleSaveEmotion = useCallback(async () => {
    if (!hasFace || !liveness?.is_live || !results?.emotion) return
    setSaveStatus('SAVING')
    try {
      await saveEmotion({
        dominant_emotion: results.emotion.dominant_emotion,
        confidence: results.emotion.confidence,
        emotion_scores: results.emotion.emotion_scores ?? null,
        entropy: results.emotion.entropy ?? null,
      })
      setSaveStatus('SAVED')
      setTimeout(() => setSaveStatus('IDLE'), 2000)
    } catch (_error) {
      setSaveStatus('ERROR')
      setTimeout(() => setSaveStatus('IDLE'), 2000)
    }
  }, [hasFace, liveness?.is_live, results])

const EMOTION_ADJECTIVES = {
  Happiness: 'Happy',
  Sadness:   'Sad',
  Anger:     'Angry',
  Fear:      'Scared',
  Disgust:   'Disgusted',
  Surprise:  'Surprised',
  Contempt:  'Contemptuous',
  Neutral:   'Neutral'

}

const emotionAdjective = EMOTION_ADJECTIVES[emotion] ?? emotion


  return (
    <div className="flex flex-col lg:flex-row gap-6 p-6 min-h-[calc(100vh-80px)] bg-surface-0 bg-cyber-grid font-body">
      
      {/* left panel */}
      <div className="flex-1 flex flex-col items-center">
        <div className="mb-4 flex items-center justify-between w-full max-w-[640px]">
          <h2 className="text-xl font-display font-bold text-purple-200 tracking-widest flex items-center gap-2">
            <Activity className={`w-5 h-5 ${isConnected ? 'text-neon-purple animate-pulse' : 'text-red-500'}`} />
            OPTICAL SENSOR FEED
          </h2>
          <div className="flex gap-2">
            <span className={`font-mono text-xs px-2 py-1 border ${isConnected ? 'border-neon-purple bg-purple-900 text-neon-purple shadow-neon-sm' : 'border-red-900 bg-red-950 text-red-500'}`}>
              {isConnected ? 'UPLINK_ESTABLISHED' : 'LINK_SEVERED'}
            </span>
            <span className="font-mono text-xs px-2 py-1 border border-purple-700 bg-surface-2 text-purple-400">
              640x480 @ 10FPS
            </span>
            <button
              onClick={() => setHeatmapEnabled(!heatmapEnabled)}
              className="px-3 py-1 text-xs border border-purple-700 bg-black text-purple-300 hover:bg-purple-900"
            >
              {heatmapEnabled ? 'Disable Heatmap' : 'Enable Heatmap'}
            </button>
          </div>
        </div>

        {/* CYBERPUNK WEB CAM CONTAINER */}
        <div className="relative w-full max-w-[640px] aspect-video bg-surface-1 border-2 border-purple-700 shadow-neon-md overflow-hidden rounded-sm group">
          <div className="absolute top-2 left-1/2 -translate-x-1/2
        text-[10px] font-mono text-purple-400 tracking-widest opacity-70">

                AI VISION PIPELINE ACTIVE

        </div>
          {error ? (
            <div className="absolute inset-0 flex items-center justify-center text-red-500 font-mono text-center p-4 bg-red-950/20">
              <AlertTriangle className="w-8 h-8 mb-2 mx-auto animate-pulse" />
              <p className="tracking-widest">{error}</p>
            </div>
            
          ) : (


            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className="w-full h-full object-cover opacity-85 mix-blend-screen grayscale-[20%] contrast-125"
            />
          )} 

          {/* AI GRID OVERLAY */}
          <div className="absolute inset-0 pointer-events-none opacity-20 z-10
          bg-[linear-gradient(rgba(170,0,255,0.2)_1px,transparent_1px),
          linear-gradient(90deg,rgba(170,0,255,0.2)_1px,transparent_1px)]
          bg-[size:40px_40px]">
          </div>
          
          {/* SCANNING LINE */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none z-20">
            <div className="scan-line"></div>
          </div>
          
          {/* CRT Scanline Overlay */}
          <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(170,0,255,0.03),rgba(0,255,0,0.01),rgba(170,0,255,0.03))] bg-[length:100%_4px,3px_100%] z-10 opacity-50"></div>
          
          {/* REACTIVE Bounding Box */}
          {hasFace && (
            <div 
              className={`absolute z-20 border-2 transition-all duration-75 ease-linear ${liveness?.is_live ? 'border-neon-purple shadow-neon-sm' : 'border-red-500 shadow-[0_0_15px_rgba(239,68,68,0.8)]'}`}
              style={bboxStyles}
            >
              {/* CORNERS */}
              <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-purple-100"></div>
              <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-purple-100"></div>
              <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-purple-100"></div>
              <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-purple-100"></div>
              
              <div className={`absolute -top-6 left-0 bg-surface-1/90 border font-mono text-[10px] px-2 py-0.5 whitespace-nowrap tracking-wider ${liveness?.is_live ? 'border-neon-purple text-neon-purple' : 'border-red-500 text-red-500'}`}>
                {liveness?.is_live ? 'ID_LOCKED' : 'SPOOF_WARNING'}
              </div>
            </div>
          )}

          {heatmapEnabled && bbox && (
            <div
              className="absolute pointer-events-none z-10"
              style={{
                ...bboxStyles,
                background: 'radial-gradient(circle, rgba(255,0,80,0.5) 0%, rgba(255,0,80,0.2) 40%, transparent 70%)',
                filter: 'blur(20px)',
              }}
            />
          )}
        </div>
          
          {/* AI OBSERVABILITY DASHBOARD */}
          <div className="w-full max-w-[640px] mt-6 border border-purple-800 bg-surface-1 p-4 shadow-[inset_0_0_20px_rgba(170,0,255,0.15)] relative">

            <div className="absolute top-0 right-0 w-24 h-24 bg-neon-purple/10 blur-2xl rounded-full"></div>

            <div className="text-purple-400 font-mono text-xs mb-4 tracking-widest flex items-center gap-2">
              <Cpu className="w-3 h-3"/> AI OBSERVABILITY
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">

              {/* SYSTEM PIPELINE */}
              <div className="bg-surface-2 border border-purple-900 p-3 font-mono text-xs">
                <div className="text-purple-400 mb-2">SYSTEM PIPELINE</div>

                <div className="flex justify-between">
                  <span>FACE DETECTED</span>
                  <span className={pipeline?.face_detected ? "text-green-400" : "text-purple-700"}>
                    {pipeline?.face_detected ? "✓" : "--"}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span>LIVENESS</span>
                  <span className={pipeline?.liveness === "LIVE" ? "text-green-400" : "text-red-500"}>
                    {pipeline?.liveness === "LIVE" ? "PASS" : "FAIL"}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span>BIOMETRIC</span>
                  <span className="text-purple-300">
                    {pipeline?.biometric_match ?? "--"}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span>EMOTION</span>
                  <span className="text-purple-300">
                    {pipeline?.emotion ?? "--"}
                  </span>
                </div>
              </div>

              {/* INFERENCE METRICS */}
              <div className="bg-surface-2 border border-purple-900 p-3 font-mono text-xs">
                <div className="text-purple-400 mb-2">INFERENCE METRICS</div>

                <div className="flex justify-between">
                  <span>FACE DETECTION</span>
                  <span className={metrics?.face_detection_ms < 50 ? "text-green-400" : metrics?.face_detection_ms < 100 ? "text-yellow-400" : "text-red-500"}>
                    {metrics?.face_detection_ms ?? "--"} ms
                  </span>
                </div>

                <div className="flex justify-between">
                  <span>LIVENESS</span>
                  <span className={metrics?.liveness_ms < 80 ? "text-green-400" : metrics?.liveness_ms < 150 ? "text-yellow-400" : "text-red-500"}>
                    {metrics?.liveness_ms ?? "--"} ms
                  </span>
                </div>

                <div className="flex justify-between">
                  <span>EMOTION</span>
                  <span className={metrics?.emotion_ms < 100 ? "text-green-400" : metrics?.emotion_ms < 200 ? "text-yellow-400" : "text-red-500"}>
                    {metrics?.emotion_ms ?? "--"} ms
                  </span>
                </div>

                <div className="flex justify-between">
                  <span>EMBEDDING</span>
                  <span className={metrics?.embedding_ms < 50 ? "text-green-400" : metrics?.embedding_ms < 100 ? "text-yellow-400" : "text-red-500"}>
                    {metrics?.embedding_ms ?? "--"} ms
                  </span>
                </div>

              </div>
              
              {/* CLIENT PERFORMANCE */}
              <div className="bg-surface-2 border border-purple-900 p-3 font-mono text-xs">
                <div className="text-purple-400 mb-2">CLIENT PERFORMANCE</div>

                <div className="flex justify-between">
                  <span>FPS</span>
                  <span className={fps > 20 ? "text-green-400" : "text-yellow-400"}>
                    {fps}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span>WS LATENCY</span>
                  <span className={latency < 100 ? "text-green-400" : latency < 300 ? "text-yellow-400" : "text-red-500"}>
                    {latency} ms
                  </span>
                </div>

                <div className="flex justify-between">
                  <span>THROUGHPUT</span>
                  <span className={throughput > 8 ? "text-green-400" : throughput > 5 ? "text-yellow-400" : "text-red-500"}>
                    {throughput} fps
                  </span>
                </div>
              </div>

              <div className="bg-surface-2 border border-purple-900 p-3 font-mono text-xs">
                <div className="text-purple-400 mb-2">EMOTION PROBABILITIES</div>

                {sortedEmotionEntries.length === 0 && (
                  <div className="text-purple-700">No emotion probabilities yet</div>
                )}

                {sortedEmotionEntries.map(([emotionKey, value]) => (
                  <div key={emotionKey} className="mb-1">
                    <div className="flex justify-between">
                      <span>{emotionKey.toUpperCase()}</span>
                      <span>{(value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-black border border-purple-900 h-2">
                      <div
                        className="bg-purple-500 h-2"
                        style={{ width: `${value * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="bg-surface-2 border border-purple-900 p-3 font-mono text-xs h-40 overflow-y-auto">
                <div className="text-purple-400 mb-2">AI EVENT STREAM</div>

                {events.map((event, i) => (
                  <div key={i} className="flex gap-2 text-purple-300">
                    <span className="text-purple-500">[{event.time}]</span>
                    <span>{event.message}</span>
                  </div>
                ))}
              </div>

              <div className="bg-surface-2 border border-purple-900 p-3">
                <ConfidentRadar entropy={entropy} />
              </div>

            </div>

          </div>
      </div>

      {/* RIGHT PANEL TELE MESSURES */}
      <div className="w-full lg:w-96 flex flex-col gap-4">
        
        {/* BLOCK 1 FOREMOST EMOTION */}
        <div className="bg-surface-1 border border-purple-800 p-4 shadow-[inset_0_0_20px_rgba(74,0,128,0.15)] relative overflow-hidden">
          <div className="absolute top-0 right-0 w-16 h-16 bg-neon-purple/5 rounded-full blur-xl"></div>
          <div className="text-purple-400 font-mono text-xs mb-1 flex items-center gap-2">
            <Cpu className="w-3 h-3" /> DOMINANT NEURAL STATE
          </div>
          <div className={`font-display text-4xl font-black uppercase tracking-wider ${hasFace ? 'text-neon-purple text-shadow-neon' : 'text-purple-700'}`}>
            {emotion}
          </div>
        </div>

        {/* BLOCK 2,  SAVE EMOTION */}
        <div style={{
          background: 'rgba(19,0,32,0.9)',
          border: '1px solid rgba(170,0,255,0.2)',
          padding: '1rem 1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
          position: 'relative',
        }}>
          {/* Corner decorations */}
          <div style={{ position: 'absolute', top: 0, left: 0, width: 10, height: 10,
            borderTop: '1px solid rgba(170,0,255,0.4)',
            borderLeft: '1px solid rgba(170,0,255,0.4)' }} />
          <div style={{ position: 'absolute', bottom: 0, right: 0, width: 10, height: 10,
            borderBottom: '1px solid rgba(170,0,255,0.4)',
            borderRight: '1px solid rgba(170,0,255,0.4)' }} />

          {/* "You are feeling" text */}
          <div style={{
            fontFamily: 'Share Tech Mono, monospace',
            fontSize: '0.95rem !important',
            letterSpacing: '0.20em',
            color: hasFace && liveness?.is_live
              ? 'rgba(170,0,255,0.7)'
              : 'rgba(170,0,255,0.25)',
            textAlign: 'center',
          }}>
            {hasFace && liveness?.is_live
              ? <>YOU ARE FEELING <span style={{
                  color: '#c69ad6ff',
                  fontWeight: 700,
                  textShadow: '0 0 8px rgba(179, 73, 233, 0.5)',
                }}>{emotionAdjective}</span></>
              : 'AWAITING LIVE FACE...'}
          </div>

          {/* SAVE EMOTION button */}
          <button
            onClick={handleSaveEmotion}
            disabled={!hasFace || !liveness?.is_live || saveStatus === 'SAVING'}
            style={{
              width: '100%',
              padding: '0.7rem',
              fontFamily: 'Orbitron, monospace',
              fontSize: '0.65rem',
              fontWeight: 700,
              letterSpacing: '0.2em',
              textTransform: 'uppercase',
              cursor: (!hasFace || !liveness?.is_live || saveStatus === 'SAVING')
                ? 'not-allowed'
                : 'pointer',
              transition: 'all 0.2s',
              border: '1px solid',
              // Color changes based on save state
              background: saveStatus === 'SAVED'  ? 'rgba(0,255,136,0.1)'
                        : saveStatus === 'ERROR'  ? 'rgba(255,0,80,0.1)'
                        : saveStatus === 'SAVING' ? 'rgba(170,0,255,0.1)'
                        : hasFace && liveness?.is_live
                          ? 'linear-gradient(135deg, rgba(102,0,179,0.6), rgba(170,0,255,0.6))'
                          : 'rgba(170,0,255,0.05)',
              borderColor: saveStatus === 'SAVED'  ? 'rgba(0,255,136,0.5)'
                        : saveStatus === 'ERROR'  ? 'rgba(255,0,80,0.5)'
                        : saveStatus === 'SAVING' ? 'rgba(170,0,255,0.3)'
                        : hasFace && liveness?.is_live
                          ? 'rgba(170,0,255,0.6)'
                          : 'rgba(170,0,255,0.15)',
              color: saveStatus === 'SAVED'  ? '#00ff88'
                  : saveStatus === 'ERROR'  ? '#ff4466'
                  : saveStatus === 'SAVING' ? 'rgba(240,204,255,0.5)'
                  : hasFace && liveness?.is_live
                    ? '#f0ccff'
                    : 'rgba(170,0,255,0.25)',
              boxShadow: saveStatus === 'SAVED' ? '0 0 16px rgba(0,255,136,0.2)'
                      : hasFace && liveness?.is_live && saveStatus === 'IDLE'
                        ? '0 0 16px rgba(170,0,255,0.2)'
                        : 'none',
            }}
          >
            {saveStatus === 'SAVING' ? 'SAVING...'
          : saveStatus === 'SAVED'  ? 'SAVED'
          : saveStatus === 'ERROR'  ? 'ERROR - RETRY'
          : hasFace && liveness?.is_live ? 'READY - SAVE EMOTION' : 'SAVE EMOTION' }
          </button>
        </div>

        {/* BLOCK 2,  RADAR */}
        <div className="h-64">
          <EmotionRadar emotionScores={emotionScores} />
        </div>

        {/* BLOCK 3 - ANTI SPOOFING*/}
        <div className="bg-surface-1 border border-purple-800 p-4 relative">
          {/* SIDE ACCENT */}
          <div className={`absolute left-0 top-0 bottom-0 w-1 ${hasFace ? (liveness?.is_live ? 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.8)]' : 'bg-red-500 animate-pulse shadow-[0_0_10px_rgba(239,68,68,0.8)]') : 'bg-purple-900'}`}></div>
          
          <div className="text-purple-400 font-mono text-xs mb-2 pl-2">BIOMETRIC INTEGRITY</div>
          <div className="flex items-center gap-4 pl-2">
            {liveness?.is_live ? (
              <ShieldCheck className="w-8 h-8 text-green-500 drop-shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
            ) : (
              <ShieldAlert className={`w-8 h-8 ${hasFace ? 'text-red-500 drop-shadow-[0_0_8px_rgba(239,68,68,0.8)]' : 'text-purple-800'}`} />
            )}
            <div className="flex-1">
              <div className={`font-mono text-sm font-bold ${hasFace ? (liveness?.is_live ? 'text-green-400' : 'text-red-500') : 'text-purple-500'}`}>
                {hasFace ? (liveness?.is_live ? 'VITAL SIGNS: REAL' : 'SPOOF DETECTED') : 'AWAITING SCAN'}
              </div>
              <div className="w-full bg-surface-3 h-1.5 mt-2 rounded-full overflow-hidden">
                <div 
                  className={`h-full transition-all duration-300 ${hasFace && liveness?.score ? (liveness.score > 0.7 ? 'bg-green-500' : liveness.score > 0.5 ? 'bg-yellow-500' : 'bg-red-500') : 'bg-red-500'}`}
                  style={{ width: `${hasFace && liveness?.score ? liveness.score * 100 : 0}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* BLOCK 4  FACE GEOMETRY AND TIRENESS LEVEL */}
        <div className="bg-surface-1 border border-purple-800 p-4">
          <div className="text-purple-400 font-mono text-xs mb-3 flex items-center gap-2">
            <Eye className="w-3 h-3" /> SPATIAL & OCULAR METRICS
          </div>
          <div className="grid grid-cols-2 gap-4">
            {/* EYEs */}
            <div className="flex flex-col gap-1 border-r border-purple-800/50 pr-2">
              <span className="text-purple-500 font-mono text-[10px]">EYE STATE</span>
              <span className={`font-mono text-sm font-bold ${isDrowsy ? 'text-red-500 animate-pulse' : 'text-neon-purple'}`}>
                {hasFace ? (isDrowsy ? 'DROWSY WARNING' : eyeState?.toUpperCase()) : '--'}
              </span>
            </div>
            {/* HEAD */}
            <div className="flex flex-col gap-1 pl-2">
              <span className="text-purple-500 font-mono text-[10px]">HEAD POSE</span>
              <span className="font-mono text-sm font-bold text-purple-200">
                {hasFace ? headPose?.toUpperCase() : '--'}
              </span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default LiveStream;
