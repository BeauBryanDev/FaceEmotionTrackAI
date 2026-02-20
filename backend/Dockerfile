# Usamos una imagen ligera de Python 3.12 (o 3.11 si prefieres)
FROM python:3.12-slim

# Establecemos variables de entorno para que Python no genere .pyc y los logs sean instantáneos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Instalamos dependencias del sistema necesarias para compilar ciertas librerías si hace falta
# Nota: Como usamos opencv-python-headless, nos ahorramos instalar libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos e instalamos dependencias primero (aprovecha la caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# Exponemos el puerto de FastAPI
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
