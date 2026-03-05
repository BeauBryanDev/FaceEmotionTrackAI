# Project Structure 
Files and Directories tree structure

## Root Directory 
```
├── backend/
├── CLAUDE.md
├── docker-compose.yml
├── frontend/
├── LICENSE
├── logs
├── notes.txt
├── privates
├── README.md
├── requirements.txt
├── scripts
└── tree.md
```
## Backend Structure 
cd backend/
```
├── alembic/
│   ├── env.py
│   ├── README
│   ├── script.py.mako
│   └── versions/
├── alembic.ini
├── app/
│   ├── api/
│   ├── core/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   ├── schemas/
│   ├── services/
│   └── utils/
├── Dockerfile
├── faceEmotions/ // human face pictures for testing scripts
├── logs/
│   └── app.log
├── ml_weights
│   ├── detection/
│   ├── emotion/
│   ├── get_models.py
│   ├── liveness/
│   └── recognition/
├── model_smoke_test.py
├── requirements.txt
├── scripts/
│   ├── db_exec.sql
│   ├── db_test.py
│   ├── db_test_script.py
│   └── emotion_db_script.py
└── tests/
    ├── conftest.py
    ├── factories.py
    ├── __init__.py
    ├── mocks.py
    ├── __pycache__
    ├── test_face_geometry.py
    ├── test_face_math.py
    ├── test_global.py
    ├── test_health.py
    ├── test_inference_engine.py
    ├── test_integration_auth.py
    ├── test_integration_emotions.py
    └── test_integration_users.py
```
## ML Model download from their repositories
cd ml_weights/ 
```
ml_weights/
├── detection
│   └── det_500m.onnx
├── emotion
│   └── emotieff_b0.onnx
├── get_models.py
├── liveness
│   └── minifasnet_v2.onnx
└── recognition
    └── w600k_mbf.onnx
```
## Alembic Migrations 
cd alembic/
```
├── env.py
├── README
├── script.py.mako
└── versions/¿
    ├── 8f225089f7e7_add_is_superuser_to_users.py
    ├── a63ab6c8f35d_initial_migration_create_users.py
    ├── bb9779230ae7_add_emotion_table.py
    └── ea31fedf1b49_add_face_session_embeddings.py
```
## Python Test Scripts
cd scripts/
```
├── db_exec.sql
├── db_test.py
├── db_test_script.py
└── emotion_db_script.py
```
## Main Logic for bakcend Inference and Endpoints
cd app/
```
├── api/
│   ├── dependencies.py
│   ├── __init__.py
│   ├── routers/
│   │   ├── analytics.py
│   │   ├── auth.py
│   │   ├── emotions.py
│   │   ├── inference.py
│   │   ├── __init__.py
│   │   └── users.py
│   └── websockets/
│       ├── __init__.py
│       ├── manager.py
│       └── stream.py
├── core/
│   ├── config.py
│   ├── database.py
│   ├── __init__.py
│   ├── logging.py
│   ├── security.py
│   └── session.py
├── __init__.py
├── main.py
├── models/
│   ├── emotions.py
│   ├── face_session.py
│   ├── __init__.py
│   └── users.py
├── schemas/
│   ├── emotion_schema.py
│   ├── __init__.py
│   ├── token_schema.py
│   └── user_schema.py
├── services/
│   ├── analytics.py
│   ├── FACE_GEOMETRY.md
│   ├── face_geometry.py
│   ├── face_math.py
│   ├── inference_engine.py
│   └── __init__.py
└── utils/
    ├── image_helper.py
    └── image_processing.py
```
##  App Tests Directory
Unit Testing and Integration Tests
cd tests/
```
.
├── conftest.py
├── factories.py
├── __init__.py
├── mocks.py
├── test_face_geometry.py
├── test_face_math.py
├── test_global.py
├── test_health.py
├── test_inference_engine.py
├── test_integration_auth.py
├── test_integration_emotions.py
└── test_integration_users.py
```
cd ../
cd frontend/
## FrontEnd 
Project UI/UX
```
.
├── dist
├── Dockerfile
├── index.html
├── node_modules
├── package.json
├── package-lock.json
├── postcss.config.js
├── src
├── tailwind.config.js
└── vite.config.js
```
---
ignore node_modules ReactJS + Axios + Tailwind + Vite
---
cd src/
## Source Directory
Main FrontEnd Logic 
```
.
├── api
│   ├── analytics.js
│   ├── auth.js
│   ├── axios.js
│   ├── emotions.js
│   ├── faces.js
│   ├── history.js
│   ├── inference.js
│   ├── interceptor.js
│   └── users.js
├── App.jsx
├── assets
│   ├── alient_face.svg
│   ├── angry_face.svg
│   ├── aquarius.svg
│   ├── emotions.svg
│   ├── emotitrack.svg
│   ├── heart.svg
│   ├── main_icon.svg
│   └── soleil.svg
├── components
│   ├── common
│   │   ├── Footer.jsx
│   │   ├── Header.jsx
│   │   ├── NavBar.jsx
│   │   └── Sidebar.jsx
│   ├── EmotionRadar.jsx
│   ├── LiveStream.jsx
│   └── ui
│       ├── AvatarCard.jsx
│       ├── Button.jsx
│       ├── GoBack.jsx
│       ├── Input.jsx
│       ├── Label.jsx
│       ├── Output.jsx
│       └── Text.jsx
├── context
│   ├── AuthContext.jsx
│   └── Biometrics.jsx
├── debugging_wb_console.js
├── hooks
│   └── useFaceTracking.js
├── index.css
├── layouts
│   ├── AuthLayout.jsx
│   ├── DashboardLayout.jsx
│   └── FaceLayout.jsx
├── main.jsx
├── notes.txt
├── pages
│   ├── Analytics.jsx
│   ├── Dashboard.jsx
│   ├── Emotions.jsx
│   ├── GhostFaces.jsx
│   ├── History.jsx
│   ├── Inference.jsx
│   ├── Login.jsx
│   ├── Profile.jsx
│   ├── Register.jsx
│   └── Users.jsx
└── services
```

