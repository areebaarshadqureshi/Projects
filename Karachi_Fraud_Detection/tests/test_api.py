"""
Pytest suite for the Fraud Detection API.

Run with: pytest tests/test_api.py -v
Requires the API server to be running: uvicorn api.main:app --reload
(or: docker-compose up)

The previous version of this file used print statements and returned
True/False without asserting anything, so `pytest tests/` reported every
test as passing even when the API was completely broken. This version
uses real assertions so failures are caught.
"""

from datetime import date

import pytest
import requests

BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session", autouse=True)
def _ensure_server_running():
    """Skip the whole module with a clear message if the API isn't up."""
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        pytest.skip(
            "API server not reachable at http://localhost:8000. "
            "Start it with: uvicorn api.main:app --reload (or docker-compose up)"
        )


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["artifacts_loaded"] is True
    assert 0.0 <= data["threshold"] <= 1.0


def test_single_prediction():
    listing = {
        "price": 12500000,
        "area_sqft": 1800,
        "bedrooms": 3,
        "baths": 2,
        "property_type": "House",
        "location": "DHA Phase 6",
        "date_added": str(date.today()),
    }
    response = requests.post(f"{BASE_URL}/predict", json=listing)
    assert response.status_code == 200

    result = response.json()
    assert 0.0 <= result["anomaly_probability"] <= 1.0
    assert isinstance(result["is_anomaly"], bool)
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert isinstance(result["top_risk_factors"], list)
    assert len(result["top_risk_factors"]) >= 1
    assert "caveat" in result and len(result["caveat"]) > 0


def test_batch_prediction():
    listings = [
        {
            "price": 12500000,
            "area_sqft": 1800,
            "bedrooms": 3,
            "baths": 2,
            "property_type": "House",
            "location": "DHA Phase 6",
            "date_added": str(date.today()),
        },
        {
            "price": 500000000,
            "area_sqft": 1000,
            "bedrooms": 2,
            "baths": 1,
            "property_type": "Flat",
            "location": "Gulshan-e-Iqbal",
            "date_added": str(date.today()),
        },
        {
            "price": 5000000,
            "area_sqft": 1200,
            "bedrooms": 2,
            "baths": 2,
            "property_type": "Flat",
            "location": "Clifton",
            "date_added": str(date.today()),
        },
    ]
    response = requests.post(f"{BASE_URL}/predict/batch", json={"listings": listings})
    assert response.status_code == 200

    result = response.json()
    assert result["total"] == len(listings)
    assert len(result["predictions"]) == len(listings)
    assert 0.0 <= result["anomaly_rate"] <= 1.0
    for pred in result["predictions"]:
        assert pred["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_model_info():
    response = requests.get(f"{BASE_URL}/model/info")
    assert response.status_code == 200

    info = response.json()
    assert info["model_name"] == "fraud_detector_v1"
    assert info["feature_count"] > 0
    assert isinstance(info["hyperparameters"], dict)
    assert isinstance(info["metrics"], dict)


def test_locations():
    response = requests.get(f"{BASE_URL}/model/locations")
    assert response.status_code == 200

    data = response.json()
    assert data["count"] == len(data["locations"])
    assert data["count"] > 0


# --- Edge cases / validation ---

def test_rejects_zero_price():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={
            "price": 0,
            "area_sqft": 1800,
            "bedrooms": 3,
            "baths": 2,
            "property_type": "House",
            "location": "DHA Phase 6",
            "date_added": str(date.today()),
        },
    )
    assert response.status_code == 422


def test_rejects_invalid_property_type():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={
            "price": 10000000,
            "area_sqft": 1800,
            "bedrooms": 3,
            "baths": 2,
            "property_type": "Mansion",  # not in the allowed Literal set
            "location": "DHA Phase 6",
            "date_added": str(date.today()),
        },
    )
    assert response.status_code == 422


def test_unknown_location_falls_back_gracefully():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={
            "price": 10000000,
            "area_sqft": 1800,
            "bedrooms": 3,
            "baths": 2,
            "property_type": "House",
            "location": "Unknown Area XYZ",
            "date_added": str(date.today()),
        },
    )
    # Unknown locations should fall back to global statistics, not error out
    assert response.status_code == 200
    result = response.json()
    assert 0.0 <= result["anomaly_probability"] <= 1.0


def test_extreme_price_flagged_high_risk():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={
            "price": 1_000_000_000,  # 1 billion PKR for a single room
            "area_sqft": 500,
            "bedrooms": 1,
            "baths": 1,
            "property_type": "Room",
            "location": "Gulshan-e-Iqbal",
            "date_added": str(date.today()),
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["risk_level"] in {"HIGH", "CRITICAL"}


def test_batch_rejects_more_than_500_listings():
    base_listing = {
        "price": 10000000,
        "area_sqft": 1500,
        "bedrooms": 3,
        "baths": 2,
        "property_type": "House",
        "location": "DHA Phase 6",
        "date_added": str(date.today()),
    }
    listings = [base_listing] * 501
    response = requests.post(f"{BASE_URL}/predict/batch", json={"listings": listings})
    assert response.status_code == 422
