import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuth } from '../context/AuthContext'
import { getStreamUrl } from '../api/inference'
import { INFERENCE_FRAME } from '../config/inference'

export const useFaceTracking = () => {
  const { token } = useAuth()

  const [results, setResults] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState(null)
  const [videoReady, setVideoReady] = useState(false)
  const [fps, setFps] = useState(0)
  const [latency, setLatency] = useState(0)
  const [throughput, setThroughput] = useState(0)
  const [emotionScores, setEmotionScores] = useState({})
  const [events, setEvents] = useState([])

  const videoRef = useRef(null)
  const wsRef = useRef(null)
  const streamRef = useRef(null)
  const canvasRef = useRef(document.createElement('canvas'))
  const waitingRef = useRef(false)
  const mountedRef = useRef(true)
  const latestResultRef = useRef(null)
  const hasPendingResultRef = useRef(false)

  const lastPingRef = useRef(0)
  const sentFramesRef = useRef(0)
  const lastFpsTimeRef = useRef(performance.now())
  const processedFramesRef = useRef(0)
  const lastThroughputTimeRef = useRef(performance.now())
  const latencyEmaRef = useRef(null)
  const throughputEmaRef = useRef(null)

  const safeSet = useCallback((setter, value) => {
    if (!mountedRef.current) return
    setter(value)
  }, [])

  const pushEvent = useCallback((message) => {
    safeSet(setEvents, (prev) => [
      { time: new Date().toLocaleTimeString(), message },
      ...prev.slice(0, 20),
    ])
  }, [safeSet])

  const stopCamera = useCallback(() => {
    const video = videoRef.current
    if (video) {
      video.oncanplay = null
      video.onloadedmetadata = null
      video.onplaying = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (video && video.srcObject) {
      video.srcObject = null
    }
    safeSet(setVideoReady, false)
  }, [safeSet])

  const startCamera = useCallback(async () => {
    try {
      safeSet(setError, null)
      safeSet(setVideoReady, false)
      waitingRef.current = false
      stopCamera()

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: 10, max: 10 } },
      })

      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }

      streamRef.current = stream
      const video = videoRef.current
      if (!video) return

      const markReady = () => safeSet(setVideoReady, true)
      video.onloadedmetadata = markReady
      video.oncanplay = markReady
      video.onplaying = markReady
      video.srcObject = stream
      await video.play().catch(() => {})
      if (video.readyState >= 2) safeSet(setVideoReady, true)
    } catch (err) {
      console.error('Camera error:', err)
      safeSet(setError, 'CAMERA ACCESS DENIED. CHECK BROWSER PERMISSIONS.')
    }
  }, [safeSet, stopCamera])

  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false
      stopCamera()
      const ws = wsRef.current
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close()
      }
      wsRef.current = null
    }
  }, [stopCamera])

  useEffect(() => {
    if (!token) {
      safeSet(setIsConnected, false)
      return
    }

    const ws = new WebSocket(getStreamUrl(token))
    wsRef.current = ws
    let cancelled = false

    ws.onopen = () => {
      if (cancelled) return
      sentFramesRef.current = 0
      processedFramesRef.current = 0
      lastFpsTimeRef.current = performance.now()
      lastThroughputTimeRef.current = performance.now()
      latencyEmaRef.current = null
      throughputEmaRef.current = null
      safeSet(setFps, 0)
      safeSet(setThroughput, 0)
      safeSet(setLatency, 0)
      safeSet(setEmotionScores, {})
      safeSet(setEvents, [])
      safeSet(setIsConnected, true)
      safeSet(setError, null)
    }

    ws.onmessage = (event) => {
      if (cancelled) return

      waitingRef.current = false

      const now = performance.now()
      if (lastPingRef.current > 0) {
        const instantLatency = now - lastPingRef.current
        const LATENCY_ALPHA = 0.2
        latencyEmaRef.current = latencyEmaRef.current == null
          ? instantLatency
          : (LATENCY_ALPHA * instantLatency) + ((1 - LATENCY_ALPHA) * latencyEmaRef.current)
        safeSet(setLatency, Math.round(latencyEmaRef.current))
      }

      processedFramesRef.current += 1
      const throughputElapsed = now - lastThroughputTimeRef.current
      if (throughputElapsed >= 1000) {
        const instantThroughput = (processedFramesRef.current * 1000) / throughputElapsed
        const THROUGHPUT_ALPHA = 0.25
        throughputEmaRef.current = throughputEmaRef.current == null
          ? instantThroughput
          : (THROUGHPUT_ALPHA * instantThroughput) + ((1 - THROUGHPUT_ALPHA) * throughputEmaRef.current)
        safeSet(setThroughput, Math.round(throughputEmaRef.current))
        processedFramesRef.current = 0
        lastThroughputTimeRef.current = now
      }

      try {
        const data = JSON.parse(event.data)
        latestResultRef.current = data
        hasPendingResultRef.current = true

        if (data.status === 'no_face_detected') {
          pushEvent('No face detected')
        }

        if (data.liveness?.is_live === true) {
          pushEvent('Liveness PASS')
        } else if (data.liveness?.is_live === false) {
          pushEvent('Liveness FAIL')
        }

        if (data.biometrics?.is_match === true) {
          pushEvent('Biometric match confirmed')
        }

        if (data.emotion) {
          let topEmotion = null
          const emotionScoresMap = data.emotion?.emotion_scores
          if (emotionScoresMap && typeof emotionScoresMap === 'object') {
            const sorted = Object.entries(emotionScoresMap).sort((a, b) => b[1] - a[1])
            topEmotion = sorted[0]?.[0] ?? null
          } else {
            const numericEmotionEntries = Object.entries(data.emotion).filter(([, value]) => typeof value === 'number')
            const sorted = numericEmotionEntries.sort((a, b) => b[1] - a[1])
            topEmotion = sorted[0]?.[0] ?? data.emotion?.dominant_emotion ?? null
          }

          if (topEmotion) {
            pushEvent(`Emotion detected: ${topEmotion}`)
          }
        }
      } catch {
        safeSet(setError, 'INVALID STREAM PAYLOAD')
      }
    }

    ws.onclose = (event) => {
      if (cancelled) return
      safeSet(setIsConnected, false)
      waitingRef.current = false
      safeSet(setFps, 0)
      safeSet(setThroughput, 0)
      if (event.code === 1008) {
        safeSet(setError, 'AUTHENTICATION FAILED. TOKEN INVALID OR EXPIRED.')
      }
    }

    ws.onerror = () => {
      if (cancelled) return
      safeSet(setError, 'WEBSOCKET CONNECTION ERROR. BACKEND UNREACHABLE.')
    }

    return () => {
      cancelled = true
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
      if (wsRef.current === ws) {
        wsRef.current = null
      }
    }
  }, [token, safeSet])

  useEffect(() => {
    const UI_COMMIT_MS = 150
    const uiTimer = setInterval(() => {
      if (!hasPendingResultRef.current) return
      hasPendingResultRef.current = false
      const latest = latestResultRef.current
      safeSet(setResults, latest)

      let nextScores = {}
      const emotionPayload = latest?.emotion
      if (emotionPayload && typeof emotionPayload === 'object') {
        if (emotionPayload.emotion_scores && typeof emotionPayload.emotion_scores === 'object') {
          nextScores = emotionPayload.emotion_scores
        } else {
          nextScores = Object.fromEntries(
            Object.entries(emotionPayload).filter(([, value]) => typeof value === 'number')
          )
        }
      }
      safeSet(setEmotionScores, nextScores)
    }, UI_COMMIT_MS)

    return () => clearInterval(uiTimer)
  }, [safeSet])

  useEffect(() => {
    if (!isConnected || !videoReady) return

    const intervalMs = INFERENCE_FRAME.intervalMs
    let encoding = false
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const interval = setInterval(() => {
      if (waitingRef.current || encoding) return

      const ws = wsRef.current
      const video = videoRef.current

      if (
        ws &&
        ws.readyState === WebSocket.OPEN &&
        video &&
        video.readyState >= 2 &&
        video.videoWidth > 0 &&
        video.videoHeight > 0
      ) {
        const TARGET_WIDTH = INFERENCE_FRAME.width
        const TARGET_HEIGHT = INFERENCE_FRAME.height

        canvas.width = TARGET_WIDTH
        canvas.height = TARGET_HEIGHT
        ctx.drawImage(video, 0, 0, TARGET_WIDTH, TARGET_HEIGHT)

        encoding = true

        canvas.toBlob((blob) => {
          encoding = false
          if (!blob) return
          if (ws.readyState !== WebSocket.OPEN) return

          waitingRef.current = true
          lastPingRef.current = performance.now()
          ws.send(blob)

          sentFramesRef.current += 1
          const now = performance.now()
          const fpsElapsed = now - lastFpsTimeRef.current
          if (fpsElapsed >= 1000) {
            const currentFps = Math.round((sentFramesRef.current * 1000) / fpsElapsed)
            safeSet(setFps, currentFps)
            sentFramesRef.current = 0
            lastFpsTimeRef.current = now
          }
        }, 'image/jpeg', INFERENCE_FRAME.jpegQuality)
      }
    }, intervalMs)

    return () => {
      clearInterval(interval)
      waitingRef.current = false
      encoding = false
    }
  }, [isConnected, videoReady, safeSet])

  return {
    videoRef,
    results,
    isConnected,
    error,
    videoReady,
    startCamera,
    stopCamera,
    fps,
    latency,
    throughput,
    emotionScores,
    events,
  }
}
