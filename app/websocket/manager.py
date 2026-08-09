"""WebSocket fan-out for live risk updates.

One call can have several watchers — the victim's app, an analyst console, a
supervisor dashboard — so updates are broadcast per call id. A dead socket is
dropped rather than allowed to break the broadcast for everyone else.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, call_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[call_id].add(websocket)

    def disconnect(self, call_id: str, websocket: WebSocket) -> None:
        self._connections[call_id].discard(websocket)
        if not self._connections[call_id]:
            self._connections.pop(call_id, None)

    def watchers(self, call_id: str) -> int:
        return len(self._connections.get(call_id, ()))

    async def broadcast(self, call_id: str, payload: dict) -> None:
        dead: list[WebSocket] = []
        for websocket in list(self._connections.get(call_id, ())):
            try:
                await websocket.send_json(payload)
            except Exception as exc:  # pragma: no cover - transport errors
                logger.debug("dropping dead websocket for %s: %s", call_id, exc)
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(call_id, websocket)
