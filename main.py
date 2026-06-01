"""Render-compatible ASGI entrypoint.

The application lives in backend.main. This file keeps older Render start
commands such as `uvicorn main:app` working.
"""

from backend.main import app

