# Frontend — FaceEmotionTrackAI

## Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18.2.0 | UI framework |
| Vite | 5.0.8 | Build tool and dev server |
| React Router | 6.22.0 | Client-side routing |
| Tailwind CSS | 3.4.19 | Utility-first styling |
| Axios | 1.6.7 | HTTP client |
| Recharts | 3.8.0 | 2D charts (radar, line, bar, pie) |
| Three.js | 0.183.2 | 3D rendering engine |
| @react-three/fiber | 8.18.0 | React wrapper for Three.js |
| @react-three/drei | 9.122.0 | Three.js helpers (OrbitControls, etc.) |
| lucide-react | 0.576.0 | Icon library |

Visual theme: **Purple Cyberpunk** — dark backgrounds, neon purple/green accents, monospace terminal aesthetics throughout.

---

## Project Structure

```
frontend/src/
├── App.jsx                    # Route definitions + auth guard
├── main.jsx                   # React entry point
│
├── api/                       # Axios API wrappers
│   └── emotions.js            # All /api/v1/emotions/* calls
│
├── components/                # Reusable UI components (28+)
│   ├── LiveStream.jsx         # Main WebSocket video stream player
│   ├── EmotionRadar.jsx       # 8-axis radar chart
│   ├── EmotionTimeline.jsx    # Dominant emotion timeline
│   ├── RussellCircumplexChart.jsx  # Valence/arousal 2D scatter
│   ├── RussellQuadrants.jsx   # Quadrant classification display
│   ├── StreamMetricsHUD.jsx   # Live ML timing overlay on video
│   ├── ExpressionSignalsHUD.jsx    # Smile/talk/attention display
│   ├── NeuralStabilityMeter.jsx    # Entropy-based stability
│   ├── PcaScatterPlot.jsx     # 2D SVG PCA scatter
│   ├── SentimentDoughnut.jsx  # Sentiment doughnut chart
│   ├── SessionEmbeddingManager.jsx # Capture + manage session embeddings
│   ├── EmotionDistributionPie.jsx
│   ├── EmotionHistoryHistogram.jsx
│   ├── EntropyTrendChart.jsx
│   ├── EmotionIntelligencePanel.jsx
│   ├── EmotionActivityTimeline.jsx
│   ├── EmotionTemporalSignalGraph.jsx
│   ├── EmotionDrift.jsx
│   ├── EmotionMomentum.jsx
│   ├── EmotionTurbulence.jsx
│   ├── EmotionVolatility.jsx
│   ├── EmotionPhaseSpace.jsx
│   ├── EmotionalTrajectory.jsx
│   ├── EmotionVectorField.jsx
│   ├── EmotionIntensityMeter.jsx
│   ├── ModelUncertaintyMeter.jsx
│   ├── PredictionStability.jsx
│   ├── common/                # Header, shared layout elements
│   ├── pca/                   # 3D PCA components (see below)
│   └── ui/                    # EmotionFlowField, EmotionVectorRadar,
│                              #   EmotionalIntelligencePanel, EmotionalPhaseSpace
│
├── pages/                     # Page-level components (15)
│   ├── About.jsx              # /about — landing/home
│   ├── Login.jsx              # /login
│   ├── Register.jsx           # /register
│   ├── Dashboard.jsx          # /dashboard
│   ├── Inference.jsx          # /inference — live stream wrapper
│   ├── Emotions.jsx           # /emotions — emotion dashboard
│   ├── EmotionReport.jsx      # /report — detailed analysis
│   ├── EmotionsAnalysis.jsx   # /russelquadrants — Russell space
│   ├── History.jsx            # /history — paginated records
│   ├── Analytics.jsx          # /analytics — 2D PCA scatter
│   ├── PCAAnalytics.jsx       # /pcaanalytics — 3D Three.js PCA
│   ├── Profile.jsx            # /profile — user settings + biometrics
│   └── Home.jsx               # (alias for About)
│
├── hooks/
│   ├── useFaceTracking.js     # WebSocket stream + metrics
│   └── usePCAData.js          # PCA data fetching
│
├── context/
│   └── AuthContext.jsx        # isAuthenticated, token, login/logout
│
├── core/affective/
│   ├── emotionDynamics.js     # Velocity, acceleration in Russell space
│   ├── emotionMetrics.js      # Energy, momentum, turbulence, regime
│   └── emotionFlowField.js    # Flow field computation
│
├── utils/
│   ├── russellMapping.js      # 8-emotion → Russell (valence, arousal)
│   └── emotionDynamics.js     # Temporal dynamics helpers
│
├── config/
│   └── inference.js           # Frame size, JPEG quality, interval
│
└── styles/
    ├── index.css              # Global Tailwind imports + custom vars
    └── pca-dashbaord.css      # PCA dashboard specific styles
```

---

## Routing — `App.jsx`

Routes are split into public (redirect to `/about` if already logged in) and protected (redirect to `/login` if not authenticated).

| Route | Page | Auth Required |
|-------|------|:---:|
| `/login` | Login.jsx | No |
| `/register` | Register.jsx | No |
| `/about` | About.jsx | No |
| `/dashboard` | Dashboard.jsx | Yes |
| `/inference` | Inference.jsx | Yes |
| `/emotions` | Emotions.jsx | Yes |
| `/report` | EmotionReport.jsx | Yes |
| `/russelquadrants` | EmotionsAnalysis.jsx | Yes |
| `/history` | History.jsx | Yes |
| `/analytics` | Analytics.jsx | Yes |
| `/pcaanalytics` | PCAAnalytics.jsx | Yes |
| `/profile` | Profile.jsx | Yes |
| `/` | Redirect → `/about` | No |

---

## Core Hook — `useFaceTracking.js`

The central hook for the live inference page. Manages the WebSocket connection, camera stream, frame sending loop, and metric tracking.

### State exposed

| State | Type | Description |
|-------|------|-------------|
| `results` | object | Latest parsed WebSocket response |
| `isConnected` | bool | WebSocket connection status |
| `error` | string | Error message if any |
| `videoReady` | bool | Camera stream initialized |
| `fps` | number | Frames sent per second |
| `latency` | number | Round-trip ms (EMA smoothed, α=0.2) |
| `throughput` | number | Frames processed/s (EMA smoothed, α=0.25) |
| `emotionScores` | object | Current `{emotion: probability}` map |
| `events` | array | Last 20 event log entries |

### Functions exposed

| Function | Description |
|----------|-------------|
| `startCamera()` | Opens `getUserMedia({video: {width:640, height:480, frameRate:10}})` |
| `stopCamera()` | Stops all media tracks, clears video src |
| `sendFrame()` | Draws canvas 320×240, encodes JPEG 0.5, sends binary blob |

### Frame sending loop

```
setInterval(sendFrame, 300ms)
    │
    ├─ Draw video frame to 320×240 canvas
    ├─ canvas.toBlob("image/jpeg", 0.5)
    ├─ websocket.send(blob)         ← binary JPEG
    ├─ Record send timestamp
    └─ On response:
        ├─ Compute latency = now - sendTime (EMA)
        ├─ Increment throughput counter (EMA per 1000ms)
        ├─ Parse JSON → set results state
        └─ Update emotionScores
```

UI state updates are debounced 150 ms to prevent excessive re-renders.

### Frame configuration — `config/inference.js`

```javascript
export const INFERENCE_FRAME = {
  width: 320,         // Canvas capture width (px)
  height: 240,        // Canvas capture height (px)
  jpegQuality: 0.5,   // Compression (0–1)
  intervalMs: 300     // Send interval (~3 FPS)
}
```

---

## PCA Hook — `usePCAData.js`

Fetches PCA visualization data on mount. Exposes `data`, `loading`, `error`, and a `reload()` function for manual refresh.

Calls `GET /api/v1/analytics/pca?include_sessions=true&session_limit=200`.

---

## Affective Computing Utilities

### `utils/russellMapping.js` — Russell Circumflex Model

Maps the 8-class EmotiEffLib output to the Russell 2D Circumflex Model (valence × arousal space).

**Input**: array of 8 probabilities in order `[Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise]`

**Output**: `{valence, arousal}` — weighted average of emotion vectors, rounded to 4 decimals.

**Emotion vectors (Valence, Arousal):**

| Emotion | Valence | Arousal |
|---------|---------|---------|
| Happiness | +0.9 | +0.7 |
| Surprise | +0.2 | +0.9 |
| Neutral | 0.0 | 0.0 |
| Contempt | −0.5 | +0.2 |
| Disgust | −0.8 | +0.3 |
| Fear | −0.9 | +0.9 |
| Anger | −0.7 | +0.8 |
| Sadness | −0.7 | −0.5 |

---

### `core/affective/emotionMetrics.js` — Emotion Metric Functions

| Function | Inputs | Output | Description |
|----------|--------|--------|-------------|
| `emotionalEnergy(valence, arousal)` | v, a | float | `0.5 × (v² + a²)` — kinetic energy magnitude |
| `emotionalMomentum(energy, velocity)` | e, v | float | `energy × velocity` |
| `emotionalDrift(current, baseline)` | `{v,a}` objects | float | Euclidean distance in Russell space |
| `emotionalTurbulence(points)` | `[{v,a}]` | float | Variance of magnitude over recent points |
| `emotionalAngularVelocity(prev, current, dt)` | two points, Δt | float | Angular change rate in atan2 space |
| `emotionalEntropy(emotions)` | string[] | float | Shannon entropy of emotion label distribution |
| `emotionalStability(volatility)` | float | float | `1 / (1 + volatility)` |
| `detectEmotionRegime({valence, arousal, volatility})` | object | string | `"chaotic"` / `"excited"` / `"depressed"` / `"agitated"` / `"calm"` |

---

### `core/affective/emotionDynamics.js` — Temporal Dynamics

| Function | Inputs | Output | Description |
|----------|--------|--------|-------------|
| `emotionalVelocity(prev, current, dt=1)` | two `{valence, arousal}` points | `{vx, vy, magnitude}` | Velocity vector in Russell space |
| `emotionalAcceleration(prevVel, vel, dt=1)` | two velocity objects | `{ax, ay, magnitude}` | Acceleration in Russell space |
| `emotionalDirection(vx, vy)` | components | float (radians) | `atan2(vy, vx)` |

---

## API Client — `api/emotions.js`

All functions use axios with the stored JWT token in the Authorization header.

| Function | Endpoint | Description |
|----------|----------|-------------|
| `getEmotionHistory(params)` | GET `/emotions/history` | Paginated history with filters |
| `getEmotionSummary()` | GET `/emotions/summary` | Aggregated emotion stats |
| `getEmotionScores(limit)` | GET `/emotions/scores` | Last N records with full distributions |
| `getEmotionDetails(emotion)` | GET `/emotions/details?emotion=X` | Breakdown of one class |
| `getEmotionScoresChart()` | GET `/emotions/scores/chart` | Histogram distribution |
| `getEmotionTimeline(limit)` | GET `/emotions/timeline` | Chronological records for charts |
| `saveEmotion(payload)` | POST `/emotions/save` | Save emotion to database |

---

## Pages

### `/inference` — Live Stream (`Inference.jsx` → `LiveStream.jsx`)

The primary real-time face analysis view.

**Layout (LiveStream.jsx):**
- Video feed with bounding box and landmark overlay (if `DEBUG_OVERLAY=true` on backend)
- `StreamMetricsHUD` — overlay showing `face_detection_ms`, `liveness_ms`, `embedding_ms`, `emotion_ms`
- `ExpressionSignalsHUD` — smile score, talk score, attention state
- EAR / MAR meters
- Liveness indicator badge (LIVE / SPOOF / UNKNOWN)
- Biometric match indicator
- Dominant emotion + confidence
- FPS / Latency / Throughput metrics
- "SAVE EMOTION" button → `POST /emotions/save`
- "CAPTURE EMBEDDING" button → `POST /analytics/session/embed`

---

### `/emotions` — Emotion Dashboard (`Emotions.jsx`)

Purple Cyberpunk dashboard showing:
- Summary stats (total detections, dominant emotion, average confidence)
- `EmotionRadar` — 8-axis Recharts radar chart
- `ConfidentRadar` — entropy-based confidence radar
- `EmotionTimeline` — scrollable dominant emotion history
- `NeuralStabilityMeter` — derived from entropy over recent records
- `EmotionDistributionPie` — Recharts pie chart

---

### `/report` — Emotion Report (`EmotionReport.jsx`)

Detailed multi-panel analysis page:
1. Total detection count header
2. Dominant emotion card + percentage breakdown
3. 8-point emotion radar chart
4. Dropdown: per-emotion detailed class breakdown
5. Aggregate score histogram bars
6. `EmotionIntelligencePanel` — velocity, turbulence, regime classification
7. Paginated detection history table (12 records/page)

Data fetched in parallel: summary, scores, history, details, chart, timeline.

---

### `/russelquadrants` — Russell Space (`EmotionsAnalysis.jsx`)

Affective science visualization:
- `RussellCircumplexChart` — 2D scatter plot of (valence, arousal) coordinates mapped from recorded emotion sessions
- `RussellQuadrants` — quadrant labels (High Arousal Positive, High Arousal Negative, Low Arousal Positive, Low Arousal Negative)
- Temporal emotion trajectory overlay
- Regime detection display (calm / excited / agitated / depressed / chaotic)

---

### `/analytics` — PCA 2D (`Analytics.jsx`)

2D principal component scatter plot (SVG, not Three.js):
- PC1 vs PC2 axes with tactical grid overlay
- Points colored by source: current user (neon green), other registered users (purple), session captures (faded)
- Animate-ping effect on current user points
- Hover tooltip showing user_id, source, PC1/PC2/PC3 values
- Statistics row: total embeddings, explained variance %, operator vector count
- Auto-scales axis bounds with 15% padding

---

### `/pcaanalytics` — PCA 3D (`PCAAnalytics.jsx`)

Interactive Three.js 3D point cloud:

**Components in `components/pca/`:**

| Component | Purpose |
|-----------|---------|
| `PCAScatter3D.jsx` | `<Canvas>` with OrbitControls, ambient + point lights |
| `PCAPoints.jsx` | Instanced mesh spheres at (x,y,z) coordinates, colored by source |
| `PCACumulativeVariance.jsx` | Cumulative variance line overlay |
| `PCACumulativeVarianceHUD.jsx` | Compact HUD showing variance numbers |
| `PCAVarianceSpectrum.jsx` | Bar chart of per-component explained variance |
| `PCAInteligencePanel.jsx` | Summary: n_components, total_variance, point counts |
| `PCALegend.jsx` | Color key for registered / session / current user points |

User can orbit, zoom, and pan the 3D scene via mouse/touch.

---

### `/history` — Emotion History (`History.jsx`)

Filterable, paginated table of all emotion records:
- Filter by emotion class (dropdown)
- Filter by date range (date pickers)
- Pagination controls
- Each row: timestamp, dominant emotion, confidence, entropy

---

### `/profile` — User Profile (`Profile.jsx`)

- Edit full_name, email, phone, age, gender, country
- Biometric verification modal before sensitive changes (`BiometricVerifyModal`)
- Face enrollment button (opens camera for biometric capture)
- Delete face embedding option
- Delete account option

---

## Development

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (hot reload at localhost:3000)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The Vite dev server proxies API calls to `localhost:8000` via `vite.config.js`. WebSocket connections go directly to `ws://localhost:8000/ws/stream`.

When running via Docker Compose, the frontend container exposes port 3000. Source code is mounted as a volume so changes hot-reload without rebuilding the image.
