"""FastAPI application for fraud detection."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

from api.schemas import (
    ListingInput,
    PredictionResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    LocationsResponse
)
from api.predict import FraudDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global detector instance
detector: FraudDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and artifacts on startup; nothing to clean up on shutdown."""
    global detector
    try:
        logger.info("Loading fraud detection model...")
        detector = FraudDetector(
            model_path="models/fraud_detector_v1.pkl",
            artifacts_path="models/artifacts"
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    # No teardown needed — model is in-memory only


# Initialize FastAPI app
app = FastAPI(
    title="Karachi Real Estate Fraud Detection API",
    # FIX: this used to say "using Random Forest" — Random Forest was one of
    # three candidates compared in NB05/NB06, but NB06's model comparison
    # selected XGBoost (tuned) as the deployed model. Wording no longer names
    # a specific algorithm here; the actual algorithm in use is always
    # available at runtime via GET /model/info (model_type field).
    description=(
        "Anomaly detection for Karachi real estate listings, trained on "
        "ensemble pseudo-labels (see /model/info for the deployed "
        "algorithm and its label caveats)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
# NOTE: allow_credentials=True cannot be combined with allow_origins=["*"]
# per the CORS spec — browsers will reject it. Since this API does not use
# cookies/auth headers, credentials are disabled here. If you need credentialed
# requests later, replace "*" with an explicit list of allowed origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML."""
    static_path = Path("static/index.html")
    if static_path.exists():
        return FileResponse(static_path)
    return HTMLResponse(
        content="<h1>Karachi Fraud Detection API</h1><p>Frontend not available. Visit <a href='/docs'>/docs</a> for API documentation.</p>",
        status_code=200
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if detector is None:
        return HealthResponse(
            status="error",
            model="none",
            version="1.0.0",
            threshold=0.0,
            artifacts_loaded=False
        )

    return HealthResponse(
        status="ok",
        model="fraud_detector_v1",
        version="1.0.0",
        threshold=detector.threshold,
        artifacts_loaded=True
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(listing: ListingInput):
    """
    Predict anomaly for a single listing.

    Returns anomaly probability, risk level, and top risk factors.
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert Pydantic model to dict
        listing_dict = listing.model_dump()

        # Get prediction
        result = detector.predict_single(listing_dict)

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """
    Predict anomalies for multiple listings (up to 500).

    Returns aggregated statistics and individual predictions.
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(request.listings) > 500:
        raise HTTPException(status_code=422, detail="Maximum batch size is 500 listings")

    try:
        # Convert Pydantic models to dicts
        listings_dicts = [listing.model_dump() for listing in request.listings]

        # Get predictions
        result = detector.predict_batch(listings_dicts)

        return BatchResponse(**result)

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get model metadata and training information.

    Returns model version, hyperparameters, metrics, and caveats.
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        info = detector.get_model_info()
        return ModelInfoResponse(**info)

    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model info: {str(e)}")


@app.get("/model/locations", response_model=LocationsResponse)
async def get_valid_locations():
    """
    Get all valid location names from training data.

    Returns the 192 Karachi locations that the model was trained on.
    Unknown locations will fall back to global statistics.
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        valid_locations = sorted(detector.feature_engineer.loc_freq_map.keys())
        return LocationsResponse(
            locations=valid_locations,
            count=len(valid_locations)
        )

    except Exception as e:
        logger.error(f"Locations retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve locations: {str(e)}")


# Mount static files (for CSS, JS, etc.)
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
