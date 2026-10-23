"""HTTP API + web interface.

    GET  /                  the web app (static files in web/)
    GET  /api/health        is a model loaded?
    GET  /api/model         model card: approach, data counts, validation metrics
    GET  /api/reports       saved experiment evidence (Approach B study, hold-out)
    POST /api/predict       multipart upload of images -> predictions
    POST /api/evaluate      (true, predicted) pairs -> confusion matrix + metrics
    POST /api/model/reload  pick up a newly copied model.keras without restarting
    GET  /docs              interactive API documentation

The model is loaded once at start-up and shared. Inference runs in a worker
thread so the event loop stays responsive while TensorFlow is busy.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from catdog import __version__
from catdog.config import ARTIFACTS_DIR, IMAGE_EXTENSIONS, LABELS, PROJECT_ROOT
from catdog.evaluation.metrics import evaluate
from catdog.server.schemas import EvaluateRequest, EvaluateResponse, HealthResponse, PredictResponse

log = logging.getLogger("catdog.server")

MODEL_DIR = Path(os.environ.get("CATDOG_MODEL_DIR", ARTIFACTS_DIR))
WEB_DIR = Path(os.environ.get("CATDOG_WEB_DIR", PROJECT_ROOT / "web"))
MAX_FILES = int(os.environ.get("CATDOG_MAX_FILES", 300))
MAX_FILE_MB = float(os.environ.get("CATDOG_MAX_FILE_MB", 30))


class ModelService:
    """Owns the Predictor and serialises access to it."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.predictor = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        from catdog.inference.predictor import Predictor

        try:
            predictor = Predictor.load(self.model_dir)
        except Exception as exc:  # surfaced in /api/health and the UI banner
            log.warning("Model not loaded: %s", exc)
            self.error = str(exc)
            return
        with self._lock:
            self.predictor, self.error = predictor, None
        log.info("Model loaded from %s", self.model_dir)

    def predict(self, items: list[tuple[str, bytes]]):
        if self.predictor is None:
            raise HTTPException(503, detail=self.error or "No model loaded")
        with self._lock:
            return self.predictor.predict_sources(items)


service = ModelService(MODEL_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(service.load)
    yield


app = FastAPI(
    title="Cat, Dog, or Neither",
    version=__version__,
    description="Sorts photographs into three piles: dog, cat, and neither.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = service.predictor is not None
    return HealthResponse(status="ok" if loaded else "no_model", model_loaded=loaded, detail=service.error, version=__version__)


@app.get("/api/model")
def model_card() -> dict:
    if service.predictor is None:
        raise HTTPException(503, detail=service.error or "No model loaded")
    info = service.predictor.info()
    info.pop("history", None)
    return info


@app.get("/api/reports")
def reports() -> dict:
    """Experiment evidence saved by training/evaluation (Approach B study, own hold-out)."""
    out: dict = {}
    for key, name in (("approach_b", "approach_b.json"), ("holdout", "holdout.json")):
        path = MODEL_DIR / "reports" / name
        if path.exists():
            data = json.loads(path.read_text())
            for heavy in ("history", "predictions"):
                data.pop(heavy, None)
            out[key] = data
    return out


@app.post("/api/model/reload", response_model=HealthResponse)
async def reload_model() -> HealthResponse:
    await run_in_threadpool(service.load)
    return health()


@app.post("/api/predict", response_model=PredictResponse)
async def predict(files: list[UploadFile] = File(...)) -> PredictResponse:
    if len(files) > MAX_FILES:
        raise HTTPException(413, detail=f"Too many files ({len(files)}). Limit is {MAX_FILES} per request.")
    items: list[tuple[str, bytes]] = []
    for upload in files:
        name = upload.filename or "upload"
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue  # quietly skip .DS_Store, .txt, etc. from dropped folders
        data = await upload.read()
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(413, detail=f"{name} is larger than {MAX_FILE_MB:.0f} MB")
        items.append((name, data))
    if not items:
        raise HTTPException(400, detail="No image files in the upload.")

    started = time.perf_counter()
    predictions = await run_in_threadpool(service.predict, items)
    counts = {label: sum(p.label == label for p in predictions) for label in (*LABELS, "error")}
    return PredictResponse(
        predictions=[p.to_dict() for p in predictions],
        counts=counts,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(body: EvaluateRequest) -> EvaluateResponse:
    report = evaluate([i.true for i in body.items], [i.predicted for i in body.items])
    errors = sum(i.predicted == "error" for i in body.items)
    return EvaluateResponse(**report.to_dict(), errors_excluded=errors)


if WEB_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
