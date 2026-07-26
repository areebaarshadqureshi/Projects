"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import date


class ListingInput(BaseModel):
    """Input schema for a single listing."""

    price: float = Field(..., gt=0, description="Property price in PKR")
    area_sqft: float = Field(..., gt=0, description="Area in square feet")
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    baths: int = Field(..., ge=0, le=20, description="Number of bathrooms")
    property_type: Literal[
        "House", "Flat", "Upper Portion", "Lower Portion",
        "Room", "Farm House", "Penthouse"
    ] = Field(..., description="Type of property")
    location: str = Field(..., min_length=1, max_length=100, description="Location in Karachi")
    date_added: date = Field(default_factory=date.today, description="Listing date")

    # Optional fields for enhanced feature engineering
    latitude: Optional[float] = Field(None, ge=24.0, le=26.0, description="Optional latitude")
    longitude: Optional[float] = Field(None, ge=66.0, le=68.0, description="Optional longitude")
    relisting_count: int = Field(0, ge=0, description="Number of times relisted (0=not relisted, higher values increase anomaly score)")

    class Config:
        json_schema_extra = {
            "example": {
                "price": 12500000,
                "area_sqft": 1800,
                "bedrooms": 3,
                "baths": 2,
                "property_type": "House",
                "location": "DHA Phase 6",
                "date_added": "2024-01-15"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    anomaly_probability: float = Field(..., ge=0, le=1, description="Probability of being anomalous")
    is_anomaly: bool = Field(..., description="Binary classification result")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(..., description="Risk category")
    risk_label: str = Field(..., description="Human-readable risk label")
    confidence_tier: str = Field(..., description="Confidence classification")
    top_risk_factors: List[str] = Field(..., description="Top features contributing to risk")
    model: str = Field(..., description="Model identifier")
    threshold: float = Field(..., description="Decision threshold used")
    caveat: str = Field(..., description="Important disclaimer about predictions")


class BatchRequest(BaseModel):
    """Request schema for batch predictions."""

    listings: List[ListingInput] = Field(..., min_length=1, max_length=500, description="List of listings")


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction results."""

    index: int
    anomaly_probability: float
    is_anomaly: bool
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_label: str


class BatchResponse(BaseModel):
    """Response schema for batch predictions."""

    total: int = Field(..., description="Total number of listings processed")
    anomalies_detected: int = Field(..., description="Number of anomalies detected")
    anomaly_rate: float = Field(..., ge=0, le=1, description="Proportion of anomalies")
    predictions: List[BatchPredictionItem] = Field(..., description="Individual predictions")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: Literal["ok", "error"] = Field(..., description="Service status")
    model: str = Field(..., description="Loaded model identifier")
    version: str = Field(..., description="API version")
    threshold: float = Field(..., description="Model decision threshold")
    artifacts_loaded: bool = Field(..., description="Whether all artifacts loaded successfully")


class ModelInfoResponse(BaseModel):
    """
    Response schema for model info endpoint.

    FIXED: previously had 'training_date' / 'hyperparameters' / 'metrics'
    fields that didn't match any key actually present in the saved model
    metadata (see predict.py FraudDetector.get_model_info), so they were
    always empty/None in practice. Replaced with the fields NB06 Section 10
    actually saves in the model card.
    """

    model_name: str
    version: str
    threshold: float
    feature_count: int
    model_type: str = Field(..., description="Algorithm selected in NB06, e.g. 'XGBoost (tuned)'")
    tuning_strategy: Optional[str] = None
    selected_by: Optional[str] = None
    best_params: dict
    cv_pr_auc_mean: Optional[float] = None
    cv_pr_auc_std: Optional[float] = None
    test_pr_auc_vs_heuristic: Optional[float] = None
    test_roc_auc_vs_heuristic: Optional[float] = None
    label_caveat: Optional[str] = None
    caveat: str


class LocationsResponse(BaseModel):
    """Response schema for valid locations endpoint."""

    locations: List[str] = Field(..., description="All valid location names from training data")
    count: int = Field(..., description="Total number of valid locations")
