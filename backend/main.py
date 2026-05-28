"""FastAPI application for FinSight."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="FinSight API", version="1.0.0")
origins = [origin.strip() for origin in os.getenv("FINSIGHT_CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AlertSubscriptionRequest(BaseModel):
    email: str
    ticker: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/bootstrap")
def bootstrap():
    try:
        from backend.dashboard_service import get_bootstrap

        return get_bootstrap()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bootstrap query failed: {exc}") from exc


@app.get("/api/dashboard")
def dashboard(
    ticker: str = Query(min_length=1, max_length=16),
    start: date = Query(),
    end: date = Query(),
):
    if start > end:
        raise HTTPException(status_code=422, detail="start must be on or before end")
    try:
        from backend.dashboard_service import build_dashboard

        return build_dashboard(ticker.upper(), start.isoformat(), end.isoformat())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dashboard query failed: {exc}") from exc


@app.post("/api/alerts/subscribe")
def subscribe_alert(request: AlertSubscriptionRequest):
    try:
        from data_ingestiion.alerts import subscribe_alert as save_subscription
        from data_ingestiion.db import get_engine

        subscription = save_subscription(get_engine(), request.email, request.ticker)
        return {"status": "subscribed", "subscription": subscription}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Alert subscription failed: {exc}") from exc


FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        requested = FRONTEND_DIST / path
        if path and requested.exists() and requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
