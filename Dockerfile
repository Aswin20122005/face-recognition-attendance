FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ATTENDANCE_DATA_DIR=/data

# libglib2.0 is needed by OpenCV's headless build
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY attendance/ attendance/

RUN useradd --create-home appuser && mkdir /data && chown appuser /data
USER appuser
VOLUME /data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "attendance.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
