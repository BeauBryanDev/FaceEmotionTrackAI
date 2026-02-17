#  Proposed Directories and Files Tree Structure
```
EmoTrack-AI/
EmoTrack-AI/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app + lifespan + CORS + WebSocket
│   │   ├── core/
│   │   │   ├── config.py            # settings (pydantic-settings)
│   │   │   ├── database.py          # SQLAlchemy + async engine
│   │   │   └── security.py          # JWT base (vacío por ahora)
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py
│   │   │   └── ws/
│   │   │       └── emotion.py       # WebSocket endpoint /ws/emotion
│   │   ├── services/
│   │   │   └── inference.py         # ONNX sessions (lazy load), predict_emotion()
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── emotion.py           # SQLModel + pgvector
│   │   │   └── schemas.py           # Pydantic response models
│   │   ├── utils/
│   │   │   └── image.py             # base64 → numpy, preprocess
│   │   └── static/                  # (opcional para docs)
│   ├── models/                      # ← aquí pegamos los .onnx (gitignored o en Docker volume)
│   │   ├── yunet.onnx
│   │   ├── emotion_enet_b0.onnx
│   │   └── arcface.onnx
│   ├── requirements.txt             # fastapi, uvicorn, onnxruntime (CPU), sqlalchemy, pgvector, python-multipart, etc.
│   ├── Dockerfile                   # python:3.12-slim + onnxruntime
│   └── alembic/                     # migrations para DB
├── tests                    
├── docker-compose.yml
├── .env.example
├── frontend/                 # React + Vite + Tailwind 
│   ├── src/
│   │   ├── components/
│   │   │   ├── WebcamFeed.tsx
│   │   │   ├── EmotionOverlay.tsx
│   │   │   └── JournalTimeline.tsx
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── public/
│   ├── tailwind.config.js
│   ├── vite.config.ts
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml        # Todo junto: postgres + backend + frontend
├── Colab/                    # Scripts para export ONNX (en tu T4)
│   └── export_model.ipynb
└── README.md
```
