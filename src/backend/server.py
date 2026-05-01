"""aiohttp application setup and WebSocket routing."""

import os
import sys

from aiohttp import web

# Add src/ to sys.path for absolute imports of the backend package
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_src_dir = os.path.join(_project_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from backend.config import DATASET_CACHE_DIR


def create_app():
    app = web.Application()

    os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

    frontend_dir = os.path.join(_project_root, "frontend")
    app.router.add_get("/", lambda req: web.FileResponse(os.path.join(frontend_dir, "index.html")))
    app.router.add_static("/static/", frontend_dir, show_index=False)

    from backend.ws_handler import ws_handler
    app.router.add_get("/ws", ws_handler)

    app.on_shutdown.append(_on_shutdown)
    return app


async def _on_shutdown(app):
    from backend.ws_handler import active_sessions
    for ws, _ in list(active_sessions.items()):
        try:
            await ws.close(code=1001, message=b"Server shutting down")
        except Exception:
            pass
    active_sessions.clear()
