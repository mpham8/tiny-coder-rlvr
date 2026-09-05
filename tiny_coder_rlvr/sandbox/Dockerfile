FROM python:3.12-slim-bookworm

WORKDIR /app

COPY tiny_coder_rlvr/sandbox/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config /app/config
COPY tiny_coder_rlvr /app/tiny_coder_rlvr

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "tiny_coder_rlvr.sandbox.runner"]
