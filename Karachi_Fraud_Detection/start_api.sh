#!/bin/bash
# Startup script for Karachi Fraud Detection API

set -e

echo "================================================"
echo "Karachi Real Estate Fraud Detection API"
echo "================================================"
echo ""

# Check if models exist
if [ ! -f "models/fraud_detector_v1.pkl" ]; then
    echo "Error: Model file not found at models/fraud_detector_v1.pkl"
    echo ""
    echo "Please ensure you have:"
    echo "  1. Run notebooks 01-06 to train the model"
    echo "  2. Model artifacts are in models/artifacts/"
    exit 1
fi

echo "✓ Model files found"
echo ""

# Check if dependencies are installed
if ! command -v uvicorn &> /dev/null; then
    echo "uvicorn not found. Installing dependencies..."
    pip install -r requirements-api.txt
    echo ""
fi

echo "✓ Dependencies ready"
echo ""

# Start the API
echo "🚀 Starting API server..."
echo ""
echo "  Frontend:  http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Health:    http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
