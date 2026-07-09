"""
Feature engineering module for fraud detection inference.
Replicates notebook 03 feature engineering logic for production use.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Union, Tuple
from datetime import datetime
from pathlib import Path


class FeatureEngineer:
    """
    Feature engineering for fraud detection inference.

    Loads all preprocessing artifacts and computes the exact 43 features
    expected by fraud_detector_v1.pkl (see model_feature_cols.pkl).
    """

    def __init__(self, artifacts_path: str = "models/artifacts"):
        """
        Load all preprocessing artifacts.

        Args:
            artifacts_path: Path to directory containing .pkl artifacts
        """
        self.artifacts_path = Path(artifacts_path)

        # Load all artifacts
        self.zscore_stats = joblib.load(self.artifacts_path / "zscore_stats_train.pkl")
        self.clip_limits = joblib.load(self.artifacts_path / "clip_limits.pkl")
        # FIXED: Use frequency encoding (loc_freq_map.pkl) instead of OrdinalEncoder
        self.loc_freq_map = joblib.load(self.artifacts_path / "loc_freq_map.pkl")
        self.market_tier_map = joblib.load(self.artifacts_path / "market_tier_map.pkl")
        self.ohe_encoder = joblib.load(self.artifacts_path / "ohe_encoder.pkl")
        self.feature_cols = joblib.load(self.artifacts_path / "model_feature_cols.pkl")
        # ADDED: Load volume stats for better weekly_volume_ratio baseline
        self.volume_stats = joblib.load(self.artifacts_path / "volume_stats_train.pkl")
        # ADDED: Load location+bedrooms median for accurate price_vs_expected_ratio
        try:
            self.loc_bed_median = joblib.load(self.artifacts_path / "loc_bed_median.pkl")
        except FileNotFoundError:
            self.loc_bed_median = None  # Graceful fallback

        # Extract global fallback values
        self.global_price_mean = self.zscore_stats['globals']['price_mean']
        self.global_price_std = self.zscore_stats['globals']['price_std']
        self.global_area_mean = self.zscore_stats['globals']['area_mean']
        self.global_area_std = self.zscore_stats['globals']['area_std']
        self.global_ppsqft_mean = self.zscore_stats['globals']['ppsqft_mean']
        self.global_ppsqft_std = self.zscore_stats['globals']['ppsqft_std']

        # ADDED: IQR global fallbacks
        self.global_price_median = self.zscore_stats['globals'].get('price_median', self.global_price_mean)
        self.global_price_iqr = self.zscore_stats['globals'].get('price_iqr', self.global_price_std * 1.35)
        self.global_ppsqft_median = self.zscore_stats['globals'].get('ppsqft_median', self.global_ppsqft_mean)
        self.global_ppsqft_iqr = self.zscore_stats['globals'].get('ppsqft_iqr', self.global_ppsqft_std * 1.35)
        self.global_area_median = self.zscore_stats['globals'].get('area_median', self.global_area_mean)
        self.global_area_iqr = self.zscore_stats['globals'].get('area_iqr', self.global_area_std * 1.35)

        # Build lookup dictionaries from z-score stats
        self._build_location_lookups()

    def _build_location_lookups(self):
        """Build fast lookup dictionaries for location-based statistics."""
        # Price location stats
        price_loc = self.zscore_stats['price_loc']
        self.price_loc_mean = dict(zip(price_loc['location'], price_loc['grp_mean']))
        self.price_loc_std = dict(zip(price_loc['location'], price_loc['grp_std']))

        # Area location stats
        area_loc = self.zscore_stats['area_loc']
        self.area_loc_mean = dict(zip(area_loc['location'], area_loc['grp_mean']))
        self.area_loc_std = dict(zip(area_loc['location'], area_loc['grp_std']))

        # Price per sqft location stats
        ppsqft_loc = self.zscore_stats['ppsqft_loc']
        self.ppsqft_loc_mean = dict(zip(ppsqft_loc['location'], ppsqft_loc['grp_mean']))
        self.ppsqft_loc_std = dict(zip(ppsqft_loc['location'], ppsqft_loc['grp_std']))

        # Area type stats (location + property_type)
        area_type = self.zscore_stats['area_type']
        self.area_type_mean = {}
        self.area_type_std = {}
        for _, row in area_type.iterrows():
            key = (row['location'], row['property_type'])
            self.area_type_mean[key] = row['grp_mean']
            self.area_type_std[key] = row['grp_std']

        # FIXED: Load actual IQR stats instead of using std * 1.35 approximation
        if 'price_iqr_loc' in self.zscore_stats:
            price_iqr_df = self.zscore_stats['price_iqr_loc']
            self.price_loc_median = dict(zip(price_iqr_df.index, price_iqr_df['price_median']))
            self.price_loc_iqr = dict(zip(price_iqr_df.index, price_iqr_df['price_iqr']))
        else:
            self.price_loc_median = {}
            self.price_loc_iqr = {}

        if 'ppsqft_iqr_loc' in self.zscore_stats:
            ppsqft_iqr_df = self.zscore_stats['ppsqft_iqr_loc']
            self.ppsqft_loc_median = dict(zip(ppsqft_iqr_df.index, ppsqft_iqr_df['ppsqft_median']))
            self.ppsqft_loc_iqr = dict(zip(ppsqft_iqr_df.index, ppsqft_iqr_df['ppsqft_iqr']))
        else:
            self.ppsqft_loc_median = {}
            self.ppsqft_loc_iqr = {}

        if 'area_iqr_loc' in self.zscore_stats:
            area_iqr_df = self.zscore_stats['area_iqr_loc']
            self.area_loc_median = dict(zip(area_iqr_df.index, area_iqr_df['area_median']))
            self.area_loc_iqr = dict(zip(area_iqr_df.index, area_iqr_df['area_iqr']))
        else:
            self.area_loc_median = {}
            self.area_loc_iqr = {}

        # Market tier lookup
        self.tier_map = dict(self.market_tier_map['tier_map'])
        self.tier_p25 = self.market_tier_map['p25']
        self.tier_p50 = self.market_tier_map['p50']
        self.tier_p75 = self.market_tier_map['p75']

    def transform(self, raw_input: Union[Dict[str, Any], pd.DataFrame],
                  return_full: bool = False) -> pd.DataFrame:
        """
        Transform raw input into model features.

        Args:
            raw_input: Single dict or DataFrame with raw listing data
            return_full: If True, return every engineered column (including
                ghost_listing_flag, geo_anomaly, price_too_low_flag,
                price_very_high_flag) instead of just the model's feature_cols.
                These flags are intentionally excluded from feature_cols (they
                are label components — see NB03), so callers that need them
                for human-readable explanations (e.g. risk-factor extraction)
                must request the full frame explicitly rather than reading
                them off the model-input frame, where they don't exist.

        Returns:
            DataFrame with the 43 model features in correct order (default),
            or the full engineered frame if return_full=True.
        """
        # Convert single dict to DataFrame
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = raw_input.copy()

        # Ensure date_added is datetime
        if 'date_added' in df.columns:
            df['date_added'] = pd.to_datetime(df['date_added'])

        # Compute all features
        df = self._add_basic_features(df)
        df = self._add_temporal_features(df)
        df = self._add_location_stats(df)
        df = self._add_zscore_features(df)
        df = self._add_iqr_features(df)
        df = self._add_price_ratio_features(df)
        df = self._add_bath_bed_features(df)
        df = self._add_market_tier(df)
        df = self._add_temporal_volume_features(df)
        df = self._add_relisting_features(df)
        df = self._add_flag_features(df)
        df = self._add_encodings(df)
        df = self._clip_features(df)

        if return_full:
            return df

        # Select final features in correct order
        return df[self.feature_cols]

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features."""
        df['price_per_sqft'] = df['price'] / df['area_sqft']
        df['bath_bed_ratio'] = df['baths'] / (df['bedrooms'] + 1e-8)
        df['area_per_bedroom'] = df['area_sqft'] / (df['bedrooms'] + 1)
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features from date_added."""
        df['listing_month'] = df['date_added'].dt.month
        df['listing_quarter'] = df['date_added'].dt.quarter
        df['listing_dow'] = df['date_added'].dt.dayofweek
        df['listing_week'] = df['date_added'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['listing_dow'] >= 5).astype(int)
        df['year_week'] = df['date_added'].dt.to_period('W').astype(str)
        return df

    def _add_location_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location-based price statistics."""
        # FIXED: Use actual saved IQR values; fall back to std-based approximation only if missing
        df['_loc_mean'] = df['location'].map(self.price_loc_mean).fillna(self.global_price_mean)
        df['_loc_std'] = df['location'].map(self.price_loc_std).fillna(self.global_price_std).replace(0, self.global_price_std)

        # Use actual IQR median if available
        if self.price_loc_median:
            df['_loc_median'] = df['location'].map(self.price_loc_median).fillna(self.global_price_median)
        else:
            df['_loc_median'] = df['_loc_mean']  # Approximation fallback

        # Use actual IQR if available
        if self.price_loc_iqr:
            df['_loc_iqr'] = df['location'].map(self.price_loc_iqr).fillna(self.global_price_iqr).replace(0, self.global_price_iqr)
        else:
            df['_loc_iqr'] = df['_loc_std'] * 1.35  # Approximation fallback

        # Price IQR deviation
        df['price_iqr_deviation'] = (df['price'] - df['_loc_median']) / df['_loc_iqr']

        # Price vs location median ratio
        df['price_vs_loc_median_ratio'] = df['price'] / df['_loc_median'].replace(0, np.nan)

        return df

    def _add_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add z-score features for price, area, and price_per_sqft."""
        # Price z-score within location
        df['_price_mean'] = df['location'].map(self.price_loc_mean).fillna(self.global_price_mean)
        df['_price_std'] = df['location'].map(self.price_loc_std).fillna(self.global_price_std).replace(0, self.global_price_std)
        df['price_loc_zscore'] = ((df['price'] - df['_price_mean']) / (df['_price_std'] + 1e-8)).fillna(0)

        # Area z-score within location
        df['_area_mean'] = df['location'].map(self.area_loc_mean).fillna(self.global_area_mean)
        df['_area_std'] = df['location'].map(self.area_loc_std).fillna(self.global_area_std).replace(0, self.global_area_std)
        df['area_sqft_loc_zscore'] = ((df['area_sqft'] - df['_area_mean']) / (df['_area_std'] + 1e-8)).fillna(0)

        # Price per sqft z-score within location
        df['_ppsqft_mean'] = df['location'].map(self.ppsqft_loc_mean).fillna(self.global_ppsqft_mean)
        df['_ppsqft_std'] = df['location'].map(self.ppsqft_loc_std).fillna(self.global_ppsqft_std).replace(0, self.global_ppsqft_std)
        df['price_per_sqft_loc_zscore'] = ((df['price_per_sqft'] - df['_ppsqft_mean']) / (df['_ppsqft_std'] + 1e-8)).fillna(0)

        # Area z-score within location + property_type
        df['_area_type_key'] = list(zip(df['location'], df['property_type']))
        df['_area_type_mean'] = df['_area_type_key'].map(self.area_type_mean).fillna(self.global_area_mean)
        df['_area_type_std'] = df['_area_type_key'].map(self.area_type_std).fillna(self.global_area_std).replace(0, self.global_area_std)
        df['area_sqft_type_zscore'] = ((df['area_sqft'] - df['_area_type_mean']) / (df['_area_type_std'] + 1e-8)).fillna(0)

        return df

    def _add_iqr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add IQR deviation features."""
        # FIXED: Use actual saved ppsqft IQR stats
        if self.ppsqft_loc_median:
            ppsqft_median = df['location'].map(self.ppsqft_loc_median).fillna(self.global_ppsqft_median)
        else:
            ppsqft_median = df['_ppsqft_mean']  # Approximation fallback

        if self.ppsqft_loc_iqr:
            ppsqft_iqr = df['location'].map(self.ppsqft_loc_iqr).fillna(self.global_ppsqft_iqr).replace(0, self.global_ppsqft_iqr)
        else:
            ppsqft_iqr = df['_ppsqft_std'] * 1.35  # Approximation fallback

        df['ppsqft_iqr_deviation'] = (df['price_per_sqft'] - ppsqft_median) / ppsqft_iqr

        # FIXED: Use actual saved area IQR stats
        if self.area_loc_median:
            area_median = df['location'].map(self.area_loc_median).fillna(self.global_area_median)
        else:
            area_median = df['_area_mean']  # Approximation fallback

        if self.area_loc_iqr:
            area_iqr = df['location'].map(self.area_loc_iqr).fillna(self.global_area_iqr).replace(0, self.global_area_iqr)
        else:
            area_iqr = df['_area_std'] * 1.35  # Approximation fallback

        df['area_iqr_deviation'] = (df['area_sqft'] - area_median) / area_iqr

        return df

    def _add_price_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price ratio and consistency features."""
        # FIXED: Use actual (location, bedrooms) joint median - matches training exactly
        if self.loc_bed_median is not None:
            df['_loc_bed_key'] = list(zip(df['location'], df['bedrooms']))
            loc_bed_dict = dict(zip(
                zip(self.loc_bed_median['location'], self.loc_bed_median['bedrooms']),
                self.loc_bed_median['_expected_price']
            ))
            df['_expected'] = df['_loc_bed_key'].map(loc_bed_dict).fillna(df['_loc_median'])
        else:
            df['_expected'] = df['_loc_median']  # Fallback to location-only median

        # Price vs expected ratio (simplified - use location median as expected)
        df['price_vs_expected_ratio'] = (df['price'] / (df['_expected'] + 1e-8)).clip(upper=50)

        # Price below/above thresholds
        df['price_below_50pct_expected'] = (df['price_vs_expected_ratio'] < 0.50).astype(int)
        df['price_above_200pct_expected'] = (df['price_vs_expected_ratio'] > 2.00).astype(int)

        # Location price CV (coefficient of variation)
        df['location_price_cv'] = df['_loc_std'] / (df['_loc_mean'] + 1e-8)

        # High CV location flag
        df['high_cv_location'] = (df['location_price_cv'] > 1.5).astype(int)

        return df

    def _add_bath_bed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bath-bedroom ratio and extreme flags."""
        df['bath_bed_extreme'] = ((df['bath_bed_ratio'] > 3.0) | (df['baths'] == 0)).astype(int)
        return df

    def _add_market_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market tier assignment."""
        # Map location to tier
        df['market_tier_derived'] = df['location'].map(self.tier_map).fillna('Middle Class')

        # Location listing count (not available in single inference - use 0)
        df['location_listing_count'] = 0

        # Location median price per sqft (use mean as proxy)
        df['location_median_ppsqft'] = df['_ppsqft_mean']

        return df

    def _add_temporal_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal volume features (set to 0 for single inference)."""
        # FIXED: Load actual avg and peak from saved artifact instead of hard-coding 0
        avg_map = self.volume_stats.get('avg_map', {})
        peak_map = self.volume_stats.get('peak_map', {})
        weeks_map = self.volume_stats.get('weeks_map', {})
        global_avg = self.volume_stats.get('global_avg_weekly', 10.0)
        global_peak = self.volume_stats.get('global_peak_weekly', 50.0)
        global_weeks = self.volume_stats.get('global_weeks_active', 20.0)

        df['avg_weekly_count'] = df['location'].map(avg_map).fillna(global_avg)
        df['peak_weekly_count'] = df['location'].map(peak_map).fillna(global_peak)
        df['total_weeks_active'] = df['location'].map(weeks_map).fillna(global_weeks)
        df['listings_per_week_in_area'] = 0  # Unknown for single inference
        # Default to 1.0 (average activity) for single inference, not 0 (below average)
        df['weekly_volume_ratio'] = 1.0

        # For batch predictions, compute these from the batch itself
        if len(df) > 1:
            week_counts = df.groupby(['location', 'year_week']).size()
            df['listings_per_week_in_area'] = df.apply(
                lambda row: week_counts.get((row['location'], row['year_week']), 0), axis=1
            )
            # FIX: weekly_volume_ratio was left at the single-inference default (1.0)
            # even after listings_per_week_in_area was recomputed from the batch above,
            # so the ratio never reflected the batch's actual volume. Recompute it here
            # the same way NB03 does: observed / average, with the same epsilon guard.
            df['weekly_volume_ratio'] = (
                df['listings_per_week_in_area'] / (df['avg_weekly_count'] + 1e-8)
            )

        return df

    def _add_relisting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relisting intensity features."""
        # For single inference, relisting_count should be provided or default to 0
        if 'relisting_count' not in df.columns:
            df['relisting_count'] = 0

        df['relisting_intensity'] = np.log1p(df['relisting_count'])
        df['is_multi_relisted'] = (df['relisting_count'] >= 10).astype(int)

        return df

    def _add_flag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add heuristic anomaly flags."""
        # CLARIFIED: ghost_listing_flag is NOT a model feature (excluded in NB03 feature selection)
        # It is replaced by relisting_count/relisting_intensity/is_multi_relisted in the actual model
        # Computing it here for completeness, but it will be dropped by feature_cols selection
        df['ghost_listing_flag'] = (df.get('relisting_count', 0) >= 5).astype(int)

        # Geo anomaly (lat/lon outside Karachi bounds)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['geo_anomaly'] = (
                (df['latitude'] < 24.74) | (df['latitude'] > 25.19) |
                (df['longitude'] < 66.68) | (df['longitude'] > 67.53)
            ).astype(int)
        else:
            df['geo_anomaly'] = 0

        # Price too low (< 1M PKR)
        df['price_too_low_flag'] = (df['price'] < 1_000_000).astype(int)

        # Price very high (> 500M PKR)
        df['price_very_high_flag'] = (df['price'] > 500_000_000).astype(int)

        return df

    def _add_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add categorical encodings."""
        # FIXED: Frequency encoding to match training (NB03 uses frequency_encode, not OrdinalEncoder)
        df['location_enc'] = df['location'].map(self.loc_freq_map).fillna(0).astype(int)

        # One-hot encode property_type and market_tier_derived
        ohe_features = self.ohe_encoder.transform(df[['property_type', 'market_tier_derived']])
        ohe_feature_names = self.ohe_encoder.get_feature_names_out(['property_type', 'market_tier_derived'])

        ohe_df = pd.DataFrame(ohe_features, columns=ohe_feature_names, index=df.index)
        df = pd.concat([df, ohe_df], axis=1)

        return df

    def _clip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature clipping using saved clip limits."""
        for col, (lower, upper) in self.clip_limits.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)
        return df
