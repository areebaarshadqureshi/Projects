# DEVELOPMENT.md

Developer guide for working with code in this repository — architecture notes, common commands, and troubleshooting.

## Project Overview

Karachi real estate anomaly detection ML system using a two-stage pseudo-label approach:
1. **Stage 1 (Unsupervised)**: Ensemble of Isolation Forest + LOF + One-Class SVM creates pseudo-labels via majority vote (6.29% suspicious rate on the full dataset)
2. **Stage 2 (Supervised)**: Random Forest (selected over XGBoost by a 0.0058 PR-AUC margin) trained on pseudo-labels, threshold optimized at 0.0247 via cross-validation

**Critical Caveat**: All predictions are based on pseudo-labels from unsupervised models, NOT verified ground truth. The final model was trained on the ensemble pseudo-label but selected/threshold-tuned against a separate heuristic label — these two label sources turned out to disagree substantially (PR-AUC 0.106, ROC-AUC 0.453 vs. the heuristic label on the test set). See README.md → "Honest Results Summary" for the full explanation. Do not assume strong real-world performance from these numbers.

## Commands

### Development
```bash
# Start API server (production mode)
make api                    # Requires model files - runs validate-models first

# Start API with auto-reload (development)
make api-dev

# Run Jupyter notebooks
make notebook               # Must run notebooks 01→06 in sequence

# Validate model files exist
make validate-models        # Checks fraud_detector_v1.pkl and artifacts/
```

### Testing
```bash
# Test API endpoints (requires running server)
make test-api              # Or: python tests/test_api.py

# Run pytest suite
make test                  # Or: pytest tests/ -v
```

### Installation
```bash
# Install API dependencies only (faster, for deployment)
make install-api           # Uses requirements-api.txt

# Install full environment (includes notebooks, analysis tools)
make install               # Uses requirements.txt

# Create conda environment
make conda-env             # From environment.yml
conda activate fraud-detection
```

### Docker
```bash
make docker-up             # Start with Docker Compose
make docker-down           # Stop containers
make docker-logs           # View logs
```

### Cleanup
```bash
make clean                 # Remove Python cache files
make clean-notebooks       # Remove .ipynb_checkpoints
make clean-all            # Clean everything including Docker
```

## Architecture

### Notebook Pipeline (Must Run in Sequence)

**01-data_cleaning.ipynb**
- Input: `data/raw/raw_listings.csv`
- Output: `data/preprocessed/all_listings_with_flags.csv`
- Filters Karachi "For Sale" listings, converts area units (Marla/Kanal → sqft), flags anomalies

**02-eda.ipynb**
- Exploratory analysis, visualizations, correlation analysis
- No artifacts generated

**03-feature_engineering.ipynb** ⚠️ CRITICAL
- Input: `all_listings_with_flags.csv`
- Output: `data/preprocessed/feature_engineered.csv` + **9 preprocessing artifacts**
- Computes 47 features, trains scalers/encoders, saves train/test splits
- **Artifacts required for production**: `scaler_robust_v1.pkl`, `loc_encoder.pkl`, `market_tier_map.pkl`, `ohe_encoder.pkl`, `clip_limits.pkl`, `zscore_stats_train.pkl`, `model_feature_cols.pkl`

**04-unsupervise_model.ipynb**
- Input: `feature_engineered.csv`
- Output: `model_scored_listings.csv` + 3 unsupervised models
- Trains Isolation Forest, LOF, OCSVM → creates `is_suspicious` pseudo-label via majority vote

**05-supervised_models.ipynb**
- Trains baseline Random Forest and XGBoost on pseudo-labels
- Initial model evaluation (~0.80 PR-AUC)

**06-model_comparison_and_hyperparameter_tuning.ipynb** ⚠️ CRITICAL
- Input: Training data with pseudo-labels
- Output: `models/fraud_detector_v1.pkl` + `models/artifacts/fraud_detector_v1_metadata.pkl` + `fraud_detector_v1_threshold.pkl`
- 150 Optuna trials per candidate model, optimizing PR-AUC vs. heuristic label via CV
- Random Forest selected over XGBoost (0.0058 PR-AUC margin)
- Finds optimal threshold (0.0247) via cross-validation, never touching the test set during search
- **Final production model** — see README.md for why the resulting test metrics are modest

### Production Inference Flow

```
Raw Input (7 fields: price, area_sqft, bedrooms, baths, property_type, location, date_added)
    ↓
FeatureEngineer (src/features/feature_engineer.py)
    - Loads preprocessing artifacts from models/artifacts/
    - Computes 43 features matching training pipeline
    - Uses location-based z-scores, IQR deviations, temporal features
    - Falls back to global statistics for unseen locations
    ↓
RobustScaler (scaler_robust_v1.pkl)
    - Scales features using training statistics
    ↓
Random Forest Model (fraud_detector_v1.pkl)
    - Predicts anomaly probability [0, 1]
    ↓
Threshold (0.0247)
    - Binary classification: anomaly if prob ≥ 0.0247
    ↓
Risk Level Mapping (scaled relative to self.threshold, NOT fixed absolute cutoffs —
see api/predict.py _map_risk_level for why):
    - CRITICAL: prob ≥ 6 × threshold
    - HIGH:     prob ≥ 3 × threshold
    - MEDIUM:   prob ≥ threshold
    - LOW:      prob < threshold

NOTE: even with this fix, a normal-looking listing and an extreme 12x-overpriced
outlier can both land in HIGH, because the model's probability range is
compressed ([~0.01, ~0.19] on the test set) and the decision threshold
(0.0247) is low enough that ~99% of listings clear it. This is a direct
downstream symptom of the train/eval label mismatch documented in README.md
"Honest Results Summary" — the risk-level bucketing logic itself now works,
but the underlying model does not separate true outliers from typical
listings as cleanly as the bucket labels might suggest. Don't oversell this
in an interview; it's worth raising proactively instead.
```

### API Structure

**api/main.py**
- FastAPI routes: `/`, `/health`, `/predict`, `/predict/batch`, `/model/info`
- Loads FraudDetector on startup
- Serves static HTML frontend

**api/predict.py**
- `FraudDetector` class: main inference engine
- Loads model + all artifacts on initialization
- Methods: `predict_single()`, `predict_batch()`, `get_model_info()`
- Extracts top risk factors from feature values

**api/schemas.py**
- Pydantic models for request/response validation
- `ListingInput`, `PredictionResponse`, `BatchRequest`, `BatchResponse`

**src/features/feature_engineer.py**
- `FeatureEngineer` class: production feature engineering
- Replicates notebook 03 logic exactly (47 features)
- Handles unseen locations with global fallback statistics
- Artifact dependencies: requires all 9 .pkl files from models/artifacts/

## Important Constraints

### Karachi-Specific Hardcoding
- **Geographic bounds**: lat [24.74, 25.19], lon [66.68, 67.53] (hardcoded in feature_engineer.py:266-268)
- **Price thresholds**: 1M-500M PKR for anomaly flags (feature_engineer.py:273-277)
- **Area conversions**: 1 Marla = 272.25 sqft, 1 Kanal = 5,445 sqft (notebook 01)
- **192 training locations**: z-score stats computed per location, unseen locations use global means
- Not transferable to other cities without retraining

### Artifact Dependencies
The API **cannot start** without these files in `models/artifacts/`:
1. `scaler_robust_v1.pkl` - RobustScaler fitted on training data
2. `loc_freq_map.pkl` - **REQUIRED** Frequency encoding for location (count of listings per location in training)
3. `market_tier_map.pkl` - Location → market tier mapping (Elite/Premium/Middle Class/Affordable)
4. `ohe_encoder.pkl` - One-hot encoder for property_type and market_tier_derived
5. `clip_limits.pkl` - Feature clipping bounds to prevent outliers
6. `zscore_stats_train.pkl` - Location-based mean/std/IQR for price, area, price_per_sqft
7. `model_feature_cols.pkl` - Ordered list of 47 feature names
8. `fraud_detector_v1_metadata.pkl` - Training metadata (date, hyperparameters, metrics)
9. `fraud_detector_v1_threshold.pkl` - Optimal decision threshold (0.3268)
10. `volume_stats_train.pkl` - **REQUIRED** Weekly volume statistics per location (avg_map, peak_map, weeks_map)
11. `loc_bed_median.pkl` - **OPTIONAL** Location+bedrooms joint median for price_vs_expected_ratio (graceful fallback if missing)

Plus the model itself: `models/fraud_detector_v1.pkl` (Random Forest classifier)

**IMPORTANT NOTES**:
- ⚠️ `loc_encoder.pkl` (old OrdinalEncoder) is **NO LONGER USED** - replaced by `loc_freq_map.pkl` for frequency encoding
- `zscore_stats_train.pkl` now contains IQR stats (`price_iqr_loc`, `ppsqft_iqr_loc`, `area_iqr_loc`) in addition to mean/std
- `volume_stats_train.pkl` must contain: `avg_map`, `peak_map`, `weeks_map`, `global_avg_weekly`, `global_peak_weekly`, `global_weeks_active`

**Regenerating artifacts**: If notebooks are re-run, ensure all artifacts are saved in notebook 03 and 06. Missing required artifacts cause `FileNotFoundError` on API startup.

### Feature Engineering Nuances

**Location Encoding** (feature_engineer.py:579)
- Uses **frequency encoding**, NOT OrdinalEncoder
- `location_enc` = count of listings for that location in training data
- Unknown locations get 0 (no training examples)
- Matches NB03 exactly: `df['location'].map(loc_freq_map).fillna(0)`

**IQR Statistics** (feature_engineer.py:137-165)
- Loads actual IQR values from `zscore_stats_train.pkl` when available
- Falls back to `std * 1.35` approximation only if IQR stats missing
- Real estate prices are highly skewed - actual IQR is more accurate than Gaussian approximation
- Keys: `price_iqr_loc`, `ppsqft_iqr_loc`, `area_iqr_loc` (DataFrames with location as index)

**Temporal Volume Features** (feature_engineer.py:230-246)
- For single predictions: `weekly_volume_ratio` defaults to **1.0** (average activity), NOT 0.0
- Loads location-specific averages from `volume_stats_train.pkl`:
  - `avg_weekly_count` - average listings per week for this location
  - `peak_weekly_count` - maximum weekly volume seen for this location
  - `total_weeks_active` - number of weeks this location had listings
- `listings_per_week_in_area` still 0 for single inference (no batch context)
- For batch predictions: computed from the batch itself (may differ from training distribution)

**Relisting Features** (feature_engineer.py:248-256)
- `relisting_count` must be provided in input or defaults to 0
- Real ghost listing detection requires historical data not available in single inference
- `relisting_intensity` = log1p(relisting_count)
- `is_multi_relisted` = 1 if relisting_count >= 10

**Location Statistics** (feature_engineer.py:137-155)
- Known locations: use pre-computed mean/std/IQR from `zscore_stats_train.pkl`
- Unknown locations: fall back to global statistics (may reduce accuracy)
- 192 locations in training set cover major Karachi areas but not exhaustive
- Price vs expected ratio uses (location, bedrooms) joint median if `loc_bed_median.pkl` available, otherwise location-only median

**Ghost Listing Flag** (feature_engineer.py:260-263)
- `ghost_listing_flag` is **NOT a model feature** (excluded in NB03 feature selection)
- Computed for reference but dropped by `df[self.feature_cols]` selection
- Actual relisting signal comes from `relisting_count`, `relisting_intensity`, `is_multi_relisted`
- These continuous features are more informative than the binary flag

### Data Drift (2018-2019 Training Data)
- Model trained on Aug 2018 - Jul 2019 listings
- Real estate prices and fraud patterns may have evolved significantly
- Listing volume statistics are 5+ years old
- Production use requires retraining on recent data

## Development Workflow

### Adding New Features
1. Modify notebook 03 to compute new features
2. Re-run notebooks 03 → 04 → 05 → 06 in sequence
3. Update `src/features/feature_engineer.py` to match notebook 03 logic exactly
4. Ensure `model_feature_cols.pkl` includes new features in correct order
5. Retrain model (notebook 06) to incorporate new features

### Updating the Model
1. Modify hyperparameter search in notebook 06
2. Re-run notebook 06 (uses existing features from notebook 03)
3. New model + threshold + metadata saved automatically
4. Restart API to load new model: `make api-dev`

### Testing Changes
1. **Notebook changes**: Run notebook with small data sample first
2. **Feature engineering**: Test `FeatureEngineer.transform()` with sample inputs
3. **API changes**: Use `make api-dev` for auto-reload, test with `make test-api`
4. **Model changes**: Compare metrics against the saved baseline in `models/artifacts/fraud_detector_v1_metadata.pkl` (test_pr_auc_vs_heuristic: 0.106, test_roc_auc_vs_heuristic: 0.453 — see README "Honest Results Summary" for why these are modest)

### Common Issues

**"Model file not found"**
- Run notebooks 01-06 in sequence first
- Check `models/fraud_detector_v1.pkl` exists
- Run `make validate-models` to diagnose

**"Artifacts not found"**
- Run notebook 03 to generate preprocessing artifacts
- Check `models/artifacts/` contains the required .pkl files (see Project Structure in README.md)

**"KeyError: feature_name"**
- Mismatch between `FeatureEngineer` and `model_feature_cols.pkl`
- Ensure `FeatureEngineer.transform()` returns the exact 43 features in correct order
- Check `self.feature_cols` matches trained model expectations

**"ValueError: feature shape mismatch"**
- Scaler expects 43 features but received different count
- Verify `FeatureEngineer` output shape: `df[self.feature_cols].shape[1]` should be 43

**Single prediction has low volume features**
- Expected behavior: temporal volume features default to 0 without batch context
- Model still works but may be less accurate for high-volume ghost listing detection

## API Usage Examples

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "price": 12500000,
    "area_sqft": 1800,
    "bedrooms": 3,
    "baths": 2,
    "property_type": "House",
    "location": "DHA Phase 6",
    "date_added": "2024-01-15"
  }'
```

Response includes:
- `anomaly_probability`: [0, 1] model confidence (note: this model's actual range is roughly [0.01, 0.19] — not calibrated, see README)
- `is_anomaly`: Boolean (prob ≥ 0.0247)
- `risk_level`: CRITICAL/HIGH/MEDIUM/LOW (scaled relative to threshold, not fixed absolute cutoffs — see api/predict.py `_map_risk_level`)
- `top_risk_factors`: Human-readable explanations (e.g., "Price is 8.3× outside typical range")
- `caveat`: Pseudo-label warning

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "listings": [
      {"price": 12500000, "area_sqft": 1800, "bedrooms": 3, ...},
      {"price": 5000000, "area_sqft": 1200, "bedrooms": 2, ...}
    ]
  }'
```

Max batch size: 500 listings

### Model Info
```bash
curl http://localhost:8000/model/info
```

Returns: model version, hyperparameters, training metrics, feature count

## Performance Notes

### Model Metrics (Test Set)

**Final model is Random Forest, not XGBoost** (selected by a 0.0058 PR-AUC margin during Optuna tuning; see `models/artifacts/fraud_detector_v1_metadata.pkl` for the authoritative source of truth on these numbers).

| Metric | vs. Heuristic Label | vs. Pseudo-Label (training target) |
|---|---|---|
| PR-AUC | 0.106 | 0.062 |
| ROC-AUC | 0.453 | not computed |
| Base rate | 0.0957 | 0.0629 |

These are modest because the model was *trained* on the ensemble pseudo-label but *selected and threshold-tuned* against a different heuristic label, and the two labels turn out to disagree substantially on what counts as anomalous (price-deviation-driven vs. relisting-driven — see NB04 Section 6 for the side-by-side feature-profile comparison). Full explanation in README.md → "Honest Results Summary". Do not quote PR-AUC 0.85 / ROC-AUC 0.92 — those numbers do not match this saved model and should not appear anywhere in this repo.

### Top 5 Feature Importance
1. `price_iqr_deviation` (15.2%) - Price deviation from location median
2. `price_loc_zscore` (12.8%) - Standardized price within location
3. `price_per_sqft` (9.3%) - Fundamental valuation metric
4. `relisting_intensity` (7.6%) - Log-transformed relisting count
5. `price_vs_expected_ratio` (6.4%) - Price vs location expected value

**Insight**: Price anomalies dominate fraud signals. Listings priced 5+ IQR deviations from location median are highly suspicious.

### Inference Speed
- Single prediction: ~50-100ms (feature engineering + scaling + Random Forest)
- Batch prediction (100 listings): ~500ms-1s
- Model load time: ~500ms (on startup only)

## Property Types
Valid values for `property_type` field (OHE encoded):
- House
- Flat
- Upper Portion
- Lower Portion
- Farm House
- Room
- Penthouse

## Market Tiers (Derived from Location)
Locations automatically mapped to tiers based on training data:
- **Elite**: DHA Phases, Clifton, Bahria Town (median 9,716 PKR/sqft)
- **Premium**: Gulshan-e-Iqbal, PECHS (median 7,521 PKR/sqft)
- **Middle Class**: North Nazimabad, FB Area (median 6,104 PKR/sqft)
- **Affordable**: Korangi, Orangi Town (median 5,089 PKR/sqft)

Unknown locations default to "Middle Class" tier.

## Testing

### Unit Tests
Run pytest on tests directory:
```bash
pytest tests/ -v
```

### API Integration Tests
Requires running server:
```bash
# Terminal 1: Start server
make api-dev

# Terminal 2: Run tests
python tests/test_api.py
```

Tests cover:
- Health check endpoint
- Single prediction with valid input
- Batch prediction
- Model info endpoint
- Invalid input validation

## When Modifying This Codebase

1. **Never modify artifacts directly** - regenerate via notebooks
2. **Keep `FeatureEngineer` in sync with notebook 03** - exact replication required
3. **Test with unseen locations** - verify global fallback statistics work
4. **Check for Karachi-specific hardcoding** - geographic bounds, price thresholds in PKR
5. **Validate against pseudo-label caveat** - all metrics are ensemble agreement, not real fraud
6. **Monitor feature count** - must be exactly 47 for model compatibility
7. **Preserve notebook sequence** - 01→06 dependencies are strict
