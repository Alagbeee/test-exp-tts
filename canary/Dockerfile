# Canary v7 — explicit startup wiring on top of v5.
FROM 15wins/canary:v5

COPY server.py /workspace/server.py
COPY start.sh /workspace/start.sh

RUN chmod +x /workspace/start.sh

ENV CANARY_MODEL_PATH=/models/canary
ENV PYTHONUNBUFFERED=1

CMD ["/workspace/start.sh"]
