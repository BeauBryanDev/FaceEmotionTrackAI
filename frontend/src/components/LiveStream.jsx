import React, { useEffect, useRef, useState } from 'react';

const LiveStream = ({ token }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const [isConnected, setIsConnected] = useState(false);
    const [analysisData, setAnalysisData] = useState(null);

    // Configuration
    const FPS = 10; // 10 frames per second is a good balance for real-time ML
    const WS_URL = `ws://localhost:8000/ws/stream?token=${token}`;

    useEffect(() => {
        let stream = null;
        let intervalId = null;

        const startCamera = async () => {
            try {
                // 1. Request access to the user's webcam
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (err) {
                console.error("Error accessing webcam: ", err);
            }
        };

        const connectWebSocket = () => {
            // 2. Establish WebSocket connection
            wsRef.current = new WebSocket(WS_URL);

            wsRef.current.onopen = () => {
                console.log("WebSocket connection established.");
                setIsConnected(true);
                
                // 3. Start capturing frames once connected
                intervalId = setInterval(captureAndSendFrame, 1000 / FPS);
            };

            wsRef.current.onmessage = (event) => {
                // 4. Receive ML analysis from FastAPI backend
                const data = JSON.parse(event.data);
                setAnalysisData(data);
            };

            wsRef.current.onerror = (error) => {
                console.error("WebSocket Error: ", error);
            };

            wsRef.current.onclose = () => {
                console.log("WebSocket connection closed.");
                setIsConnected(false);
                if (intervalId) clearInterval(intervalId);
            };
        };

        const captureAndSendFrame = () => {
            if (!videoRef.current || !canvasRef.current || wsRef.current?.readyState !== WebSocket.OPEN) {
                return;
            }

            const canvas = canvasRef.current;
            const context = canvas.getContext('2d');
            const video = videoRef.current;

            // Draw current video frame onto the canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert canvas to Base64 JPEG (quality 0.7 for faster transmission)
            const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
            
            // Strip the 'data:image/jpeg;base64,' prefix
            const base64Image = dataUrl.split(',')[1];

            // Send payload to the Python backend
            const payload = { image: base64Image };
            wsRef.current.send(JSON.stringify(payload));
        };

        // Initialize
        startCamera().then(() => connectWebSocket());

        // Cleanup function when component unmounts
        return () => {
            if (intervalId) clearInterval(intervalId);
            if (wsRef.current) wsRef.current.close();
            if (stream) stream.getTracks().forEach(track => track.stop());
        };
    }, [token]);

    return (
        <div className="flex flex-col items-center p-4 bg-gray-900 rounded-lg shadow-xl text-white">
            <h2 className="text-2xl font-bold mb-4">Live Biometric Analysis</h2>
            
            <div className="relative">
                {/* Visible Video Stream */}
                <video 
                    ref={videoRef} 
                    autoPlay 
                    playsInline 
                    muted 
                    className="rounded-lg border-2 border-blue-500 shadow-lg transform scale-x-[-1]" 
                />
                
                {/* Hidden Canvas for processing */}
                <canvas 
                    ref={canvasRef} 
                    width="640" 
                    height="480" 
                    className="hidden" 
                />

                {/* Overlay ML Results */}
                {analysisData && analysisData.emotion && (
                    <div className="absolute top-4 left-4 bg-black bg-opacity-70 p-4 rounded text-sm">
                        <p className="text-green-400 font-bold">
                            Emotion: {analysisData.emotion.dominant_emotion.toUpperCase()}
                        </p>
                        <p className="text-gray-300">
                            Confidence: {(analysisData.emotion.confidence * 100).toFixed(1)}%
                        </p>
                        {analysisData.liveness && (
                            <p className={analysisData.liveness.is_live ? "text-green-500" : "text-red-500"}>
                                Liveness: {analysisData.liveness.is_live ? "Passed" : "Failed"}
                            </p>
                        )}
                    </div>
                )}
            </div>

            <div className="mt-4">
                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${isConnected ? 'bg-green-600' : 'bg-red-600'}`}>
                    {isConnected ? "Server Connected" : "Disconnected"}
                </span>
            </div>
        </div>
    );
};

export default LiveStream;