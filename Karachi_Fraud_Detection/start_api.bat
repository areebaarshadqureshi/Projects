@echo off
REM Startup script for Karachi Fraud Detection API (Windows)

echo ================================================
echo Karachi Real Estate Fraud Detection API
echo ================================================
echo.

REM Check if models exist
if not exist "models\fraud_detector_v1.pkl" (
    echo Error: Model file not found at models\fraud_detector_v1.pkl
    echo.
    echo Please ensure you have:
    echo   1. Run notebooks 01-06 to train the model
    echo   2. Model artifacts are in models\artifacts\
    exit /b 1
)

echo [OK] Model files found
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.11+
    exit /b 1
)

echo [OK] Python found
echo.

REM Install dependencies if needed
python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements-api.txt
    echo.
)

echo [OK] Dependencies ready
echo.

REM Start the API
echo Starting API server...
echo.
echo   Frontend:  http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo   Health:    http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
