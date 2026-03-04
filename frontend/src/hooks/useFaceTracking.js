
import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuth } from '../context/AuthContext'
import { getStreamUrl } from '../api/inference'

export const useFaceTracking = () => {
  const { token } = useAuth()

  const [results,     setResults]     = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error,       setError]       = useState(null)
  const [videoReady,  setVideoReady]  = useState(false)

  const videoRef    = useRef(null)
  const wsRef       = useRef(null)
  const canvasRef   = useRef(document.createElement('canvas'))
  const rafRef      = useRef(null)

  // ---------------------------------------------------------------------------
  // 1. CAMERA
  // ---------------------------------------------------------------------------
  const startCamera = useCallback(async () => {
    try {
      setError(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: 15 } },
      })
      if (!videoRef.current) return
      videoRef.current.srcObject = stream
      videoRef.current.oncanplay = () => setVideoReady(true)
    } catch (err) {
      console.error('Camera error:', err)
      setError('CAMERA ACCESS DENIED. CHECK BROWSER PERMISSIONS.')
    }
  }, [])

  // ---------------------------------------------------------------------------
  // 2. WEBSOCKET + INFERENCE LOOP
  // Both live in the same effect so the RAF loop and the WS share the same
  // "cancelled" flag and the same "waiting" variable. No ref leakage between
  // StrictMode mount cycles.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!token || !isConnected) return
    if (!videoReady) return

    let cancelled = false
    let waiting   = false       // request-response gate: local to this effect
    let lastTime  = 0
    const INTERVAL = 150        // ms between frames (~6-7 FPS, safe for 4 ONNX models on CPU)

    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    // Override onmessage to use this effect's cancelled flag and waiting var
    ws.onmessage = (event) => {
      if (cancelled) return
      waiting = false           // ungate: ready to send next frame
      try {
        const data = JSON.parse(event.data)
        setResults(data)
      } catch {
        setError('INVALID STREAM PAYLOAD')
      }
    }

    const sendFrame = (timestamp) => {
      if (cancelled) return

      if (timestamp - lastTime >= INTERVAL && !waiting) {
        const video = videoRef.current
        if (
          video &&
          video.readyState >= 2 &&
          video.videoWidth > 0 &&
          video.videoHeight > 0 &&
          ws.readyState === WebSocket.OPEN
        ) {
          const canvas  = canvasRef.current
          canvas.width  = video.videoWidth
          canvas.height = video.videoHeight
          canvas.getContext('2d').drawImage(video, 0, 0)
          const b64 = canvas.toDataURL('image/jpeg', 0.7)
          waiting = true        // gate until onmessage resets it
          ws.send(JSON.stringify({ image: b64 }))
          lastTime = timestamp
        }
      }

      rafRef.current = requestAnimationFrame(sendFrame)
    }

    rafRef.current = requestAnimationFrame(sendFrame)

    return () => {
      cancelled = true
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }
  }, [token, isConnected, videoReady])

  // ---------------------------------------------------------------------------
  // 3. WEBSOCKET CONNECTION - one effect, one connection
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!token) {
      setIsConnected(false)
      return
    }

    let cancelled = false
    const ws = new WebSocket(getStreamUrl(token))
    wsRef.current = ws

    ws.onopen = () => {
      if (cancelled) return
      setIsConnected(true)
      setError(null)
    }

    // onmessage is overridden by the inference loop effect when active.
    // This fallback handles messages received before videoReady.
    ws.onmessage = (event) => {
      if (cancelled) return
      try {
        setResults(JSON.parse(event.data))
      } catch { /* ignore */ }
    }

    ws.onclose = (event) => {
      if (cancelled) return
      setIsConnected(false)
      if (event.code === 1008) {
        setError('AUTHENTICATION FAILED. TOKEN INVALID OR EXPIRED.')
      }
    }

    ws.onerror = () => {
      if (cancelled) return
      setError('WEBSOCKET CONNECTION ERROR. BACKEND UNREACHABLE.')
    }

    return () => {
      cancelled = true
      setIsConnected(false)
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
      if (wsRef.current === ws) wsRef.current = null
    }
  }, [token])

  // ---------------------------------------------------------------------------
  // 4. CLEANUP camera tracks on unmount
  // ---------------------------------------------------------------------------

  useEffect(() => {
  return () => {
    const video = videoRef.current
    if (video && video.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop())
      video.srcObject = null
    }
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    setVideoReady(false)  // reset for next mount
    setResults(null)      // clear results on unmount
    setIsConnected(false)
  }
}, [])
  // useEffect(() => {
  //   return () => {
  //     const video = videoRef.current
  //     if (video && video.srcObject) {
  //       video.srcObject.getTracks().forEach((t) => t.stop())
  //     }
  //     if (rafRef.current) cancelAnimationFrame(rafRef.current)
  //   }
  // }, [])


useEffect(() => {
  return () => {
    // Solo limpia recursos del DOM y timers
    // NUNCA setState aqui - el componente ya esta desmontado
    const video = videoRef.current
    if (video && video.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop())
      video.srcObject = null
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }
}, [])

  return {
    videoRef,
    results,
    isConnected,
    error,
    videoReady,
    startCamera,
  }
}
