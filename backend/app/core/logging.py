import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from app.core.config import settings

# --- Logging Configuration ---

LOG_DIR      = Path("/app/logs")
LOG_FILE     = LOG_DIR / "app.log"
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO").upper()
ENVIRONMENT  = os.getenv("ENVIRONMENT", "development")


LOG_MAX_BYTES =  10 * 1024 * 1024  # 10 MB
LOG_BACKUPS   =   5
LOG_BACKUP_COUNT =  5

# Create log directory if it doesn't exist
if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True)

# Create log file if it doesn't exist
if not LOG_FILE.exists():
    LOG_FILE.touch()

# Set log file size limit
handler = logging.handlers.RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUPS
)

# Set log file format
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class DevFormatter(logging.Formatter):
    
    LEVEL_COLORS = {
        "DEBUG"    : "\033[94m",   # BLUE
        "INFO"     : "\033[92m",   # GREEN
        "WARNING"  : "\033[93m",   # YELLOW
        "ERROR"    : "\033[91m",   # RED
        "CRITICAL" : "\033[95m",   # MAGNETA
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        
        color     = self.LEVEL_COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level     = f"{color}{record.levelname:<8}{self.RESET}"
        name      = record.name[:40]

        message = record.getMessage()

        # Si hay excepcion la agrega en la linea siguiente
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return f"{timestamp} | {level} | {name:<40} | {message}"
    
    
    
class JSONFormatter(logging.Formatter):
    
    """
    Formatter ON PRODUCTION, it produce a JSON line per event
    CloudWatch Logs Insights in order to query fields as this format

    Ejemplo de linea:
    {
        "timestamp": "2026-02-25T10:32:11.123Z",
        "level": "INFO",
        "logger": "app.services.inference_engine",
        "message": "Loaded detection model",
        "environment": "production",
        "service": "emotrack-backend"
    }
    """
    
    def format(self, record: logging.LogRecord) -> str:
        
        log_entry: dict[str, Any] = {
            "timestamp"   : datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level"       : record.levelname,
            "logger"      : record.name,
            "message"     : record.getMessage(),
            "environment" : ENVIRONMENT,
            "service"     : "emotrack-backend",
        }
        
        
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id

        if hasattr(record, "endpoint"):
            log_entry["endpoint"] = record.endpoint

        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        # Excepcion con traceback completo para CloudWatch
        if record.exc_info:
            
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


# Main SetUp 

def setup_logging() -> None:
    
    """
        Initialize the application's logging system.

    It must be called only ONCE in the lifespan of main.py before loading the ONNX models.

    Configure two handlers:

    - StreamHandler: stdout -> docker compose logs / CloudWatch

    - RotatingFileHandler: /app/logs/app.log -> volume on host machine or EFS in AWS

    """
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Elegir formatter segun entorno
    if ENVIRONMENT == "production":
        formatter = JSONFormatter()
    else:
        formatter = DevFormatter()

    # Nivel de log
    numeric_level = getattr(logging, LOG_LEVEL, logging.INFO)

    # --- Handler 1: stdout ---
    # docker compose logs -f backend / CloudWatch en ECS
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)

    # --- Handler 2: archivo rotativo ---
    # /app/logs/app.log montado como volumen ./logs en tu maquina
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename    = LOG_FILE,
            maxBytes    = LOG_MAX_BYTES,
            backupCount = LOG_BACKUP_COUNT,
            encoding    = "utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        handlers = [stream_handler, file_handler]
    except PermissionError:
        # Si el directorio no esta montado correctamente en el contenedor
        # continua solo con stdout para no romper el arranque
        handlers = [stream_handler]
        logging.warning(
            f"No se pudo crear el archivo de log en {LOG_FILE}. "
            f"Verifica el volumen en docker-compose.yml. "
            f"Continuando solo con stdout."
        )

    # Configurar root logger
    logging.basicConfig(
        level    = numeric_level,
        handlers = handlers,
        force    = True   # fuerza reemplazar handlers previos (uvicorn los setea primero)
    )

    # Silenciar loggers ruidosos de librerias externas
    # que generan mucho ruido sin aportar informacion util
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("passlib").setLevel(logging.WARNING)

    # Log de confirmacion de arranque
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging inicializado. "
        f"env={ENVIRONMENT}  level={LOG_LEVEL}  "
        f"file={LOG_FILE}"
    )
    
    
def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger con el nombre del modulo llamador.


    """
    return logging.getLogger(name)
