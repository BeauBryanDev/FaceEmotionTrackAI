from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
from datetime import datetime

from app.core.database import get_db
from app.core.logging import get_logger
from app.api.dependencies import get_current_active_user
from app.models.users import User
from app.models.emotions import Emotion

router = APIRouter()

logger = get_logger(__name__)

VALID_EMOTIONS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise"
]


# GET /api/v1/emotions/history

@router.get("/history", status_code=status.HTTP_200_OK)
async def get_emotion_history(
    page: int              = Query(default=1, ge=1, description="Numero de pagina"),
    page_size: int         = Query(default=10, ge=1, le=100, description="Registros por pagina"),
    emotion_filter: Optional[str] = Query(
        default=None,
        description="Filtrar por emocion especifica (ej: Happiness, Anger)"
    ),
    date_from: Optional[datetime] = Query(
        default=None,
        description="Filtrar desde esta fecha (ISO 8601: 2026-02-01T00:00:00)"
    ),
    date_to: Optional[datetime] = Query(
        default=None,
        description="Filtrar hasta esta fecha (ISO 8601: 2026-02-28T23:59:59)"
    ),
    current_user: User     = Depends(get_current_active_user),
    db: Session            = Depends(get_db)
) -> dict:
    """
    Retorna el historial paginado de emociones detectadas para el usuario autenticado.

    Soporta filtros opcionales por emocion y rango de fechas.
    Los resultados se ordenan del mas reciente al mas antiguo.
    """
    logger.info(
        f"Historial de emociones solicitado.",
        extra={"user_id": current_user.id}
    )

    # Validar filtro de emocion si fue provisto
    if emotion_filter and emotion_filter not in VALID_EMOTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Emocion invalida: '{emotion_filter}'. "
                   f"Valores validos: {VALID_EMOTIONS}"
        )

    # Construir query base
    query = db.query(Emotion).filter(Emotion.user_id == current_user.id)

    # Aplicar filtros opcionales
    if emotion_filter:
        query = query.filter(Emotion.dominant_emotion == emotion_filter)

    if date_from:
        query = query.filter(Emotion.timestamp >= date_from)

    if date_to:
        query = query.filter(Emotion.timestamp <= date_to)

    # Total de registros para calcular paginas
    total_records = query.count()
    total_pages   = max(1, -(-total_records // page_size))  # ceil division

    # Paginacion: ordenar del mas reciente al mas antiguo
    offset  = (page - 1) * page_size
    records = (
        query
        .order_by(Emotion.timestamp.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    logger.info(
        f"Historial retornado: {len(records)} registros de {total_records} totales.",
        extra={"user_id": current_user.id}
    )

    return {
        "user_id"      : current_user.id,
        "page"         : page,
        "page_size"    : page_size,
        "total_records": total_records,
        "total_pages"  : total_pages,
        "filters"      : {
            "emotion" : emotion_filter,
            "date_from": date_from.isoformat() if date_from else None,
            "date_to"  : date_to.isoformat() if date_to else None,
        },
        "records": [
            {
                "id"              : record.id,
                "dominant_emotion": record.dominant_emotion,
                "confidence"      : round(record.confidence, 4),
                "emotion_scores"  : record.emotion_scores,
                "timestamp"       : record.timestamp.isoformat(),
            }
            for record in records
        ]
    }


# GET /api/v1/emotions/summary
@router.get("/summary", status_code=status.HTTP_200_OK)
async def get_emotion_summary(
    current_user: User = Depends(get_current_active_user),
    db: Session        = Depends(get_db)
) -> dict:
    """
    Retorna estadisticas agregadas de emociones para el usuario autenticado.

    Incluye:
    - Conteo de detecciones por emocion
    - Emocion dominante (la mas frecuente)
    - Confianza promedio por emocion
    - Total de detecciones registradas
    """
    logger.info(
        f"Resumen de emociones solicitado.",
        extra={"user_id": current_user.id}
    )

    # Query de agregacion: conteo y confianza promedio por emocion
    # Una sola query a la DB en lugar de N queries por emocion
    aggregation = (
        db.query(
            Emotion.dominant_emotion,
            func.count(Emotion.id).label("count"),
            func.avg(Emotion.confidence).label("avg_confidence"),
        )
        .filter(Emotion.user_id == current_user.id)
        .group_by(Emotion.dominant_emotion)
        .order_by(func.count(Emotion.id).desc())
        .all()
    )

    if not aggregation:
        logger.info(
            f"Sin registros de emociones para el usuario.",
            extra={"user_id": current_user.id}
        )
        return {
            "user_id"         : current_user.id,
            "total_detections": 0,
            "dominant_emotion": None,
            "emotion_stats"   : [],
            "message"         : "No hay registros de emociones para este usuario."
        }

    total_detections = sum(row.count for row in aggregation)

    # La primera fila ya es la emocion dominante por el ORDER BY DESC
    dominant_emotion = aggregation[0].dominant_emotion

    emotion_stats = [
        {
            "emotion"       : row.dominant_emotion,
            "count"         : row.count,
            "percentage"    : round((row.count / total_detections) * 100, 2),
            "avg_confidence": round(float(row.avg_confidence), 4),
        }
        for row in aggregation
    ]

    logger.info(
        f"Resumen generado. Total: {total_detections} detecciones. "
        f"Dominante: {dominant_emotion}",
        extra={"user_id": current_user.id}
    )

    return {
        "user_id"         : current_user.id,
        "total_detections": total_detections,
        "dominant_emotion": dominant_emotion,
        "emotion_stats"   : emotion_stats,
    }


# GET /api/v1/emotions/details
@router.get("/details", status_code=status.HTTP_200_OK)
async def get_emotion_details(
    emotion: str = Query(..., description="Emocion a consultar"),
    current_user: User = Depends(get_current_active_user),
    db: Session        = Depends(get_db)
) -> dict:
    """
    Retorna detalles de una emocion especifica para el usuario autenticado.

    Incluye:
    - Emocion
    - Confianza promedio por clase
    - Clases con mayor probabilidad
    """
    logger.info(
        f"Detalles de emocion solicitado.",
        extra={"user_id": current_user.id}
    )

    # Validar filtro de emocion si fue provisto
    if emotion not in VALID_EMOTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Emocion invalida: '{emotion}'. "
                   f"Valores validos: {VALID_EMOTIONS}"
        )

    # Query de agregacion: conteo y confianza promedio por clase
    # Una sola query a la DB en lugar de N queries por clase
    aggregation = (
        db.query(
            Emotion.dominant_emotion,
            Emotion.confidence,
            Emotion.emotion_scores,
        )
        .filter(Emotion.user_id == current_user.id)
        .filter(Emotion.dominant_emotion == emotion)
        .first()
    )

    if not aggregation:
        logger.info(
            f"Sin registros de emociones para el usuario.",
            extra={"user_id": current_user.id}
        )
        return {
            "user_id"         : current_user.id,
            "emotion"         : emotion,
            "total_detections": 0,
            "dominant_emotion": None,
            "emotion_stats"   : [],
            "message"         : "No hay registros de emociones para este usuario."
        }

    logger.info(
        f"Detalles de emocion generados.",
        extra={"user_id": current_user.id}
    )

    return {
        "user_id"         : current_user.id,
        "emotion"         : emotion,
        "total_detections": 1,
        "dominant_emotion": aggregation.dominant_emotion,
        "emotion_stats"   : [
            {
                "class"       : key,
                "confidence"  : round(value, 4),
                "percentage"  : round((value / aggregation.confidence) * 100, 2),
            }
            for key, value in aggregation.emotion_scores.items()
        ],
    }   
    
    
# GET /api/v1/emotions/scores

@router.get("/scores", status_code=status.HTTP_200_OK)
async def get_emotion_scores(
    limit: int         = Query(default=10, ge=1, le=50, description="Ultimos N registros"),
    current_user: User = Depends(get_current_active_user),
    db: Session        = Depends(get_db)
) -> dict:
    """
    Retorna los ultimos N registros que tienen el campo emotion_scores poblado.

    emotion_scores contiene la distribucion completa de probabilidades
    de las 8 emociones. Util para graficas de radar o barras en el dashboard.

    NOTA: Este campo es None hasta que se implemente detect_emotion v2.
    """
    logger.info(
        f"Scores de emociones solicitados. Limit: {limit}",
        extra={"user_id": current_user.id}
    )

    # Solo retornar registros donde emotion_scores no es NULL
    records = (
        db.query(Emotion)
        .filter(
            Emotion.user_id == current_user.id,
            Emotion.emotion_scores.isnot(None)
        )
        .order_by(Emotion.timestamp.desc())
        .limit(limit)
        .all()
    )

    if not records:
        return {
            "user_id": current_user.id,
            "limit"  : limit,
            "count"  : 0,
            "records": [],
            "message": (
                "No hay registros con emotion_scores disponibles. "
                "Este campo se poblara cuando se implemente detect_emotion v2."
            )
        }

    logger.info(
        f"Scores retornados: {len(records)} registros.",
        extra={"user_id": current_user.id}
    )

    return {
        "user_id": current_user.id,
        "limit"  : limit,
        "count"  : len(records),
        "records": [
            {
                "id"              : record.id,
                "dominant_emotion": record.dominant_emotion,
                "confidence"      : round(record.confidence, 4),
                "emotion_scores"  : record.emotion_scores,
                "timestamp"       : record.timestamp.isoformat(),
            }
            for record in records
        ]
    }


# GET /api/v1/emotions/scores/chart

@router.get("/scores/chart", status_code=status.HTTP_200_OK)
async def get_emotion_scores_chart(
    limit: int         = Query(default=10, ge=1, le=50, description="Ultimos N registros"),
    current_user: User = Depends(get_current_active_user),
    db: Session        = Depends(get_db)
) -> dict:
    """
    Retorna un histograma de la distribución de las emociones detectadas.

    Incluye:
    - Histograma de la distribución de las emociones detectadas
    - Número de registros totales
    - Dominante emoción
    """
    logger.info(
        f"Histograma de emociones solicitado.",
        extra={"user_id": current_user.id}
    )

    # Query de agregacion: conteo y confianza promedio por clase
    # Una sola query a la DB en lugar de N queries por clase
    aggregation = (
        db.query(
            Emotion.dominant_emotion,
            Emotion.confidence,
            Emotion.emotion_scores,
        )
        .filter(Emotion.user_id == current_user.id)
        .filter(Emotion.emotion_scores.isnot(None))
        .first()
    )

    if not aggregation:
        logger.info(
            f"Sin registros de emociones para el usuario.",
            extra={"user_id": current_user.id}
        )
        return {
            "user_id"         : current_user.id,
            "total_detections": 0,
            "dominant_emotion": None,
            "emotion_stats"   : [],
            "message"         : "No hay registros de emociones para este usuario."
        }

    logger.info(
        f"Histograma de emociones generado.",
        extra={"user_id": current_user.id}
    )

    return {
        "user_id"         : current_user.id,
        "total_detections": 1,
        "dominant_emotion": aggregation.dominant_emotion,
        "emotion_stats"   : [
            {
                "class"       : key,
                "confidence"  : round(value, 4),
                "percentage"  : round((value / aggregation.confidence) * 100, 2),
            }
            for key, value in aggregation.emotion_scores.items()
        ],
    }   