import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuth } from '../context/AuthContext'
import { getStreamUrl } from '../api/inference'

export const useFaceTracking = () => {
  const { token } = useAuth()

  const [results, setResults] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState(null)
  const [videoReady, setVideoReady] = useState(false)

  const videoRef = useRef(null)
  const wsRef = useRef(null)
  const streamRef = useRef(null)
  const canvasRef = useRef(document.createElement('canvas'))
  const rafRef = useRef(null)
  const waitingRef = useRef(false)
  const lastSentAtRef = useRef(0)
  const mountedRef = useRef(true)

  const safeSet = useCallback((setter, value) => {
    if (!mountedRef.current) return
    setter(value)
  }, [])

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

      // Always reset previous stream before requesting a new one.
      stopCamera()

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: 15 } },
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
      await video.play().catch(() => {
        // Browser autoplay constraints can reject this promise; video may still play.
      })
      if (video.readyState >= 2) {
        safeSet(setVideoReady, true)
      }
    } catch (err) {
      console.error('Camera error:', err)
      safeSet(setError, 'CAMERA ACCESS DENIED. CHECK BROWSER PERMISSIONS.')
    }
  }, [safeSet, stopCamera])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
      stopCamera()
      const ws = wsRef.current
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close()
      }
      wsRef.current = null
    }
  }, [stopCamera])

  // WebSocket connection lifecycle.
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
      safeSet(setIsConnected, true)
      safeSet(setError, null)
    }

    ws.onmessage = (event) => {
      if (cancelled) return
      waitingRef.current = false
      try {
        safeSet(setResults, JSON.parse(event.data))
      } catch {
        safeSet(setError, 'INVALID STREAM PAYLOAD')
      }
    }

    ws.onclose = (event) => {
      if (cancelled) return
      safeSet(setIsConnected, false)
      waitingRef.current = false
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
        wsRef.current = null;
      }
    }
  }, [token, safeSet])

  // Frame capture loop.
  useEffect(() => {
    if (!isConnected || !videoReady) return

    const intervalMs = 300

    const interval = setInterval(() => {

      if (waitingRef.current) return

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
        const canvas = canvasRef.current

        const TARGET_WIDTH = 320
        const TARGET_HEIGHT = 240

        canvas.width = TARGET_WIDTH
        canvas.height = TARGET_HEIGHT

        const ctx = canvas.getContext('2d')

        if (!ctx) return

        ctx.drawImage(video, 0, 0, TARGET_WIDTH, TARGET_HEIGHT)

        waitingRef.current = true

        ws.send(
          JSON.stringify({
            image: canvas.toDataURL('image/jpeg', 0.5)
          })
        )
      }

    }, intervalMs)

    return () => {
      clearInterval(interval)
      waitingRef.current = false
    }

  }, [isConnected, videoReady])

  return {
    videoRef,
    results,
    isConnected,
    error,
    videoReady,
    startCamera,
    stopCamera,
  }
}
