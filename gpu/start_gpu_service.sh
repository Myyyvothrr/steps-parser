# only use 1 worker, multiple have not been tested with model
uvicorn parse_service:app \
  --host 0.0.0.0 \
  --port $STEPS_PARSER_PORT \
  --workers 1 \
  --app-dir "src/"
