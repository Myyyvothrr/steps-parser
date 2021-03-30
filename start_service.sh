# only use 1 worker, multiple have not been tested with model
uvicorn parse_service:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1 \
  --app-dir "src/"