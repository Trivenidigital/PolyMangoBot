"""
Production ML Opportunity Predictor
Priority 3 implementation from analysis report.

Features:
- LightGBM for fast, accurate predictions
- Comprehensive feature engineering pipeline
- Lagged features and rolling statistics
- Isotonic calibration for reliable probabilities
- Online learning with concept drift detection
- Feature importance monitoring
"""

import numpy as np
import time
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# Try to import sklearn for calibration
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("PolyMangoBot.production_ml")


@dataclass
class PredictionResult:
    """Result of ML prediction"""
    opportunity_id: str
    probability: float           # Calibrated probability of profit
    raw_probability: float       # Uncalibrated model output
    confidence: float            # Confidence in prediction
    expected_profit: float       # Expected profit given probability
    risk_score: float           # Risk assessment
    feature_contributions: Dict[str, float]  # SHAP-like contributions

    # Metadata
    model_version: str = "1.0"
    prediction_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def should_execute(self) -> bool:
        """Determine if opportunity should be executed"""
        return (
            self.probability > 0.55 and
            self.confidence > 0.4 and
            self.expected_profit > 0 and
            self.risk_score < 0.7
        )

    def to_dict(self) -> Dict:
        return {
            "opportunity_id": self.opportunity_id,
            "probability": self.probability,
            "raw_probability": self.raw_probability,
            "confidence": self.confidence,
            "expected_profit": self.expected_profit,
            "risk_score": self.risk_score,
            "should_execute": self.should_execute,
            "prediction_time_ms": self.prediction_time_ms,
            "top_features": dict(sorted(
                self.feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5])
        }


@dataclass
class TrainingExample:
    """Single training example"""
    features: np.ndarray
    label: int  # 1 = profitable, 0 = not profitable
    profit: float
    timestamp: float
    metadata: Dict = field(default_factory=dict)


class FeatureEngineeringPipeline:
    """
    Comprehensive feature engineering for arbitrage prediction.

    Creates features:
    - Price features (spread, mid, ratio)
    - Volume features (imbalance, depth)
    - Time features (hour, day, seasonality)
    - Lagged features (momentum, trends)
    - Rolling statistics (mean, std, percentiles)
    - Cross-venue features (correlation, divergence)
    - Microstructure features (OBI, toxicity)
    """

    def __init__(self):
        # Feature groups
        self.feature_names: List[str] = []

        # History for rolling features
        self._price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._profit_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Target encoding cache
        self._venue_pair_profit: Dict[str, float] = {}
        self._hour_profit: Dict[int, float] = {}

        # Scaler for normalization
        self._scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._scaler_fitted = False

    def extract_features(self, opportunity: Dict) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all features from an opportunity.

        Returns: (feature_array, feature_names)
        """
        features = []
        names = []

        # ============================================
        # BASIC PRICE FEATURES
        # ============================================
        buy_price = opportunity.get("buy_price", 0)
        sell_price = opportunity.get("sell_price", 0)
        mid_price = (buy_price + sell_price) / 2 if buy_price and sell_price else 0

        spread = sell_price - buy_price
        spread_pct = (spread / buy_price * 100) if buy_price > 0 else 0
        spread_bps = spread_pct * 100

        features.extend([
            spread,
            spread_pct,
            spread_bps,
            np.log1p(spread) if spread > 0 else 0,  # Log spread
        ])
        names.extend([
            "spread_raw", "spread_pct", "spread_bps", "spread_log"
        ])

        # ============================================
        # VOLUME FEATURES
        # ============================================
        buy_volume = opportunity.get("buy_volume", 0)
        sell_volume = opportunity.get("sell_volume", 0)
        total_volume = buy_volume + sell_volume

        volume_imbalance = (buy_volume - sell_volume) / (total_volume + 1e-10)
        tradeable_volume = min(buy_volume, sell_volume)
        volume_ratio = buy_volume / (sell_volume + 1e-10)

        features.extend([
            np.log1p(buy_volume),
            np.log1p(sell_volume),
            np.log1p(total_volume),
            volume_imbalance,
            np.log1p(tradeable_volume),
            np.clip(volume_ratio, 0.1, 10),  # Clipped ratio
        ])
        names.extend([
            "buy_volume_log", "sell_volume_log", "total_volume_log",
            "volume_imbalance", "tradeable_volume_log", "volume_ratio"
        ])

        # ============================================
        # MICROSTRUCTURE FEATURES
        # ============================================
        obi = opportunity.get("obi", 0)
        obi_5 = opportunity.get("obi_5_level", obi)
        toxicity = opportunity.get("toxicity_score", 0)
        depth_ratio = opportunity.get("depth_ratio", 1)

        features.extend([
            obi,
            obi_5,
            toxicity,
            np.log1p(depth_ratio),
            abs(obi),  # Magnitude of imbalance
        ])
        names.extend([
            "obi", "obi_5_level", "toxicity_score",
            "depth_ratio_log", "obi_magnitude"
        ])

        # ============================================
        # COST FEATURES
        # ============================================
        estimated_fees = opportunity.get("estimated_fees", 0)
        estimated_slippage = opportunity.get("estimated_slippage", 0)
        total_cost = estimated_fees + estimated_slippage

        gross_profit = opportunity.get("gross_profit", spread * tradeable_volume)
        net_profit = gross_profit - total_cost
        profit_margin = net_profit / (gross_profit + 1e-10)

        features.extend([
            estimated_fees,
            estimated_slippage,
            total_cost,
            gross_profit,
            net_profit,
            profit_margin,
        ])
        names.extend([
            "estimated_fees", "estimated_slippage", "total_cost",
            "gross_profit", "net_profit", "profit_margin"
        ])

        # ============================================
        # TIME FEATURES
        # ============================================
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        day_of_week = now.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        # Cyclical encoding for hour (captures midnight wrap-around)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Market session indicators
        is_us_session = 1 if 9 <= hour <= 16 else 0
        is_asia_session = 1 if 0 <= hour <= 8 else 0
        is_europe_session = 1 if 7 <= hour <= 15 else 0

        features.extend([
            hour,
            minute,
            day_of_week,
            is_weekend,
            hour_sin,
            hour_cos,
            is_us_session,
            is_asia_session,
            is_europe_session,
        ])
        names.extend([
            "hour", "minute", "day_of_week", "is_weekend",
            "hour_sin", "hour_cos",
            "is_us_session", "is_asia_session", "is_europe_session"
        ])

        # ============================================
        # VENUE FEATURES
        # ============================================
        buy_venue = opportunity.get("buy_venue", "")
        sell_venue = opportunity.get("sell_venue", "")

        # Venue encoding
        venue_map = {
            "polymarket": 1, "kraken": 2, "coinbase": 3,
            "binance": 4, "ftx": 5, "": 0
        }
        buy_venue_id = venue_map.get(buy_venue.lower(), 0)
        sell_venue_id = venue_map.get(sell_venue.lower(), 0)
        same_venue = 1 if buy_venue == sell_venue else 0
        venue_pair_id = buy_venue_id * 10 + sell_venue_id

        features.extend([
            buy_venue_id,
            sell_venue_id,
            same_venue,
            venue_pair_id,
        ])
        names.extend([
            "buy_venue_id", "sell_venue_id", "same_venue", "venue_pair_id"
        ])

        # ============================================
        # LAGGED FEATURES (Momentum)
        # ============================================
        market = opportunity.get("market", "unknown")
        key = f"{market}_{buy_venue}_{sell_venue}"

        # Update history
        self._spread_history[key].append(spread_pct)
        self._volume_history[key].append(total_volume)

        spread_hist = list(self._spread_history[key])
        volume_hist = list(self._volume_history[key])

        # Spread momentum
        if len(spread_hist) >= 2:
            spread_change_1 = spread_pct - spread_hist[-2]
        else:
            spread_change_1 = 0

        if len(spread_hist) >= 5:
            spread_change_5 = spread_pct - spread_hist[-5]
        else:
            spread_change_5 = 0

        # Volume trend
        if len(volume_hist) >= 5:
            volume_trend = np.mean(volume_hist[-5:]) - np.mean(volume_hist[-10:-5]) if len(volume_hist) >= 10 else 0
        else:
            volume_trend = 0

        features.extend([
            spread_change_1,
            spread_change_5,
            volume_trend,
        ])
        names.extend([
            "spread_change_1", "spread_change_5", "volume_trend"
        ])

        # ============================================
        # ROLLING STATISTICS
        # ============================================
        if len(spread_hist) >= 10:
            spread_mean_10 = np.mean(spread_hist[-10:])
            spread_std_10 = np.std(spread_hist[-10:])
            spread_zscore = (spread_pct - spread_mean_10) / (spread_std_10 + 1e-10)
            spread_percentile = np.searchsorted(sorted(spread_hist), spread_pct) / len(spread_hist)
        else:
            spread_mean_10 = spread_pct
            spread_std_10 = 0
            spread_zscore = 0
            spread_percentile = 0.5

        if len(volume_hist) >= 10:
            volume_mean_10 = np.mean(volume_hist[-10:])
            volume_ratio_to_mean = total_volume / (volume_mean_10 + 1e-10)
        else:
            volume_mean_10 = total_volume
            volume_ratio_to_mean = 1

        features.extend([
            spread_mean_10,
            spread_std_10,
            spread_zscore,
            spread_percentile,
            volume_mean_10,
            volume_ratio_to_mean,
        ])
        names.extend([
            "spread_mean_10", "spread_std_10", "spread_zscore",
            "spread_percentile", "volume_mean_10", "volume_ratio_to_mean"
        ])

        # ============================================
        # TARGET-ENCODED FEATURES
        # ============================================
        venue_pair_key = f"{buy_venue}_{sell_venue}"
        historical_profit = self._venue_pair_profit.get(venue_pair_key, 0.5)
        hour_profit = self._hour_profit.get(hour, 0.5)

        features.extend([
            historical_profit,
            hour_profit,
        ])
        names.extend([
            "venue_pair_historical_profit", "hour_historical_profit"
        ])

        # ============================================
        # INTERACTION FEATURES
        # ============================================
        spread_volume_interaction = spread_pct * np.log1p(total_volume)
        obi_spread_interaction = obi * spread_pct
        toxicity_volume_interaction = toxicity * np.log1p(total_volume)

        features.extend([
            spread_volume_interaction,
            obi_spread_interaction,
            toxicity_volume_interaction,
        ])
        names.extend([
            "spread_volume_interaction", "obi_spread_interaction",
            "toxicity_volume_interaction"
        ])

        self.feature_names = names
        return np.array(features, dtype=np.float32), names

    def update_target_encoding(
        self,
        buy_venue: str,
        sell_venue: str,
        hour: int,
        was_profitable: bool
    ):
        """Update target-encoded features with new outcome"""
        # Exponential moving average update
        alpha = 0.1

        venue_pair_key = f"{buy_venue}_{sell_venue}"
        current = self._venue_pair_profit.get(venue_pair_key, 0.5)
        self._venue_pair_profit[venue_pair_key] = alpha * (1 if was_profitable else 0) + (1 - alpha) * current

        current_hour = self._hour_profit.get(hour, 0.5)
        self._hour_profit[hour] = alpha * (1 if was_profitable else 0) + (1 - alpha) * current_hour


class ConceptDriftDetector:
    """
    Detect concept drift in model predictions.
    Triggers retraining when performance degrades.
    """

    def __init__(self, window_size: int = 100, threshold: float = 0.3):
        self.window_size = window_size
        self.threshold = threshold

        self.prediction_errors: deque = deque(maxlen=window_size)
        self.drift_detected = False
        self.last_drift_time = 0.0

    def update(self, predicted_prob: float, actual_outcome: int) -> bool:
        """
        Update with new prediction/outcome pair.
        Returns True if drift detected.
        """
        # Brier score for this prediction
        error = (predicted_prob - actual_outcome) ** 2
        self.prediction_errors.append(error)

        if len(self.prediction_errors) < self.window_size // 2:
            return False

        # Compare recent vs older performance
        mid = len(self.prediction_errors) // 2
        recent_error = np.mean(list(self.prediction_errors)[mid:])
        older_error = np.mean(list(self.prediction_errors)[:mid])

        # Drift if recent error significantly higher
        if recent_error > older_error * (1 + self.threshold):
            self.drift_detected = True
            self.last_drift_time = time.time()
            logger.warning(
                f"Concept drift detected! Recent error: {recent_error:.4f}, "
                f"Older error: {older_error:.4f}"
            )
            return True

        self.drift_detected = False
        return False

    @property
    def current_error(self) -> float:
        if not self.prediction_errors:
            return 0.0
        return np.mean(self.prediction_errors)


class FeatureImportanceTracker:
    """
    Track feature importance over time to detect regime changes.
    """

    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        self.importance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def update(self, importances: Dict[str, float]):
        """Update with new feature importances"""
        for name, importance in importances.items():
            self.importance_history[name].append(importance)

    def get_stability_report(self) -> Dict[str, Dict]:
        """
        Get stability report for all features.
        High instability may indicate regime change.
        """
        report = {}

        for name, history in self.importance_history.items():
            if len(history) < 10:
                continue

            values = list(history)
            mean = np.mean(values)
            std = np.std(values)

            # Coefficient of variation (CV)
            cv = std / (mean + 1e-10)

            # Trend
            x = np.arange(len(values))
            trend = np.polyfit(x, values, 1)[0]

            report[name] = {
                "mean": mean,
                "std": std,
                "cv": cv,
                "trend": trend,
                "is_stable": cv < 0.5  # CV < 50% considered stable
            }

        return report

    def get_unstable_features(self, cv_threshold: float = 0.5) -> List[str]:
        """Get list of features with unstable importance"""
        report = self.get_stability_report()
        return [
            name for name, stats in report.items()
            if stats["cv"] > cv_threshold
        ]


class ProductionMLPredictor:
    """
    Production-grade ML predictor using LightGBM.

    Features:
    - Fast predictions (<5ms)
    - Calibrated probabilities
    - Online learning with drift detection
    - Feature importance tracking
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_training_samples: int = 100,
        retrain_interval: int = 500
    ):
        self.model_path = model_path
        self.min_training_samples = min_training_samples
        self.retrain_interval = retrain_interval

        # Feature engineering
        self.feature_pipeline = FeatureEngineeringPipeline()

        # Model
        self.model = None
        self.calibrator = None
        self.model_version = "1.0"
        self._is_trained = False

        # Training data buffer
        self.training_buffer: List[TrainingExample] = []
        self.samples_since_retrain = 0

        # Monitoring
        self.drift_detector = ConceptDriftDetector()
        self.importance_tracker = FeatureImportanceTracker()

        # Performance tracking
        self.predictions_made = 0
        self.total_prediction_time_ms = 0.0

        # Initialize model
        if LIGHTGBM_AVAILABLE:
            self._init_model()

        # Load existing model if path provided
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def _init_model(self):
        """Initialize LightGBM model with tuned hyperparameters"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, using fallback")
            return

        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            importance_type='gain',
            verbose=-1
        )

    def add_training_example(
        self,
        opportunity: Dict,
        was_profitable: bool,
        actual_profit: float
    ):
        """Add a training example"""
        features, _ = self.feature_pipeline.extract_features(opportunity)

        example = TrainingExample(
            features=features,
            label=1 if was_profitable else 0,
            profit=actual_profit,
            timestamp=time.time(),
            metadata=opportunity
        )

        self.training_buffer.append(example)
        self.samples_since_retrain += 1

        # Update target encoding
        self.feature_pipeline.update_target_encoding(
            opportunity.get("buy_venue", ""),
            opportunity.get("sell_venue", ""),
            datetime.now().hour,
            was_profitable
        )

        # Check if should retrain
        should_retrain = (
            len(self.training_buffer) >= self.min_training_samples and
            (self.samples_since_retrain >= self.retrain_interval or
             self.drift_detector.drift_detected or
             not self._is_trained)
        )

        if should_retrain:
            self._train()

    def _train(self):
        """Train/retrain the model"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("Cannot train: LightGBM not available")
            return

        if len(self.training_buffer) < self.min_training_samples:
            logger.info(f"Not enough samples to train: {len(self.training_buffer)}/{self.min_training_samples}")
            return

        logger.info(f"Training model with {len(self.training_buffer)} samples...")

        # Prepare data
        X = np.array([ex.features for ex in self.training_buffer])
        y = np.array([ex.label for ex in self.training_buffer])

        # Handle class imbalance
        n_positive = np.sum(y)
        n_negative = len(y) - n_positive

        if n_positive == 0 or n_negative == 0:
            logger.warning("Cannot train with only one class")
            return

        scale_pos_weight = n_negative / n_positive

        # Update model parameters
        self.model.set_params(scale_pos_weight=scale_pos_weight)

        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        # Calibrate probabilities
        if SKLEARN_AVAILABLE and len(X_val) >= 20:
            try:
                self.calibrator = CalibratedClassifierCV(
                    self.model,
                    method='isotonic',
                    cv='prefit'
                )
                self.calibrator.fit(X_val, y_val)
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                self.calibrator = None

        # Update feature importance
        feature_importance = dict(zip(
            self.feature_pipeline.feature_names,
            self.model.feature_importances_
        ))
        self.importance_tracker.update(feature_importance)

        self._is_trained = True
        self.samples_since_retrain = 0
        self.model_version = f"1.{int(time.time())}"

        # Log training metrics
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        logger.info(f"Training complete. Train acc: {train_score:.3f}, Val acc: {val_score:.3f}")

        # Log top features
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        logger.info(f"Top features: {top_features}")

    def predict(self, opportunity: Dict) -> PredictionResult:
        """
        Make prediction for an opportunity.
        """
        start_time = time.perf_counter_ns()

        # Extract features
        features, feature_names = self.feature_pipeline.extract_features(opportunity)

        # Default result if model not trained
        if not self._is_trained or self.model is None:
            return PredictionResult(
                opportunity_id=opportunity.get("id", str(time.time())),
                probability=0.5,
                raw_probability=0.5,
                confidence=0.0,
                expected_profit=0.0,
                risk_score=0.5,
                feature_contributions={},
                model_version="untrained",
                prediction_time_ms=0.0
            )

        # Reshape for prediction
        X = features.reshape(1, -1)

        # Get raw probability
        raw_proba = self.model.predict_proba(X)[0, 1]

        # Get calibrated probability
        if self.calibrator is not None:
            calibrated_proba = self.calibrator.predict_proba(X)[0, 1]
        else:
            calibrated_proba = raw_proba

        # Calculate feature contributions (simplified SHAP-like)
        contributions = self._calculate_contributions(features, feature_names)

        # Calculate confidence based on prediction certainty and drift
        confidence = self._calculate_confidence(calibrated_proba)

        # Calculate expected profit
        gross_profit = opportunity.get("gross_profit", 0)
        estimated_costs = opportunity.get("estimated_costs", 0)
        net_profit = gross_profit - estimated_costs
        expected_profit = net_profit * calibrated_proba - abs(net_profit) * (1 - calibrated_proba) * 0.3

        # Risk score
        risk_score = self._calculate_risk_score(opportunity, calibrated_proba, confidence)

        # Track prediction time
        prediction_time_ms = (time.perf_counter_ns() - start_time) / 1_000_000
        self.predictions_made += 1
        self.total_prediction_time_ms += prediction_time_ms

        return PredictionResult(
            opportunity_id=opportunity.get("id", str(time.time())),
            probability=float(calibrated_proba),
            raw_probability=float(raw_proba),
            confidence=confidence,
            expected_profit=expected_profit,
            risk_score=risk_score,
            feature_contributions=contributions,
            model_version=self.model_version,
            prediction_time_ms=prediction_time_ms
        )

    def _calculate_contributions(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Calculate simplified feature contributions.
        Uses feature importance * feature value as approximation.
        """
        if not self._is_trained:
            return {}

        importances = self.model.feature_importances_
        contributions = {}

        for i, name in enumerate(feature_names):
            if i < len(importances):
                # Normalize by importance and multiply by feature value
                contrib = importances[i] * features[i]
                contributions[name] = float(contrib)

        # Normalize to sum to 1
        total = sum(abs(v) for v in contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions

    def _calculate_confidence(self, probability: float) -> float:
        """Calculate confidence in the prediction"""
        # Base confidence from prediction certainty
        certainty = abs(probability - 0.5) * 2  # 0-1, higher for extreme predictions

        # Adjust for drift
        if self.drift_detector.drift_detected:
            certainty *= 0.5

        # Adjust for sample size
        sample_factor = min(1.0, len(self.training_buffer) / 500)
        certainty *= (0.5 + 0.5 * sample_factor)

        # Adjust for recent prediction error
        error = self.drift_detector.current_error
        if error > 0.3:
            certainty *= 0.7

        return min(0.95, max(0.1, certainty))

    def _calculate_risk_score(
        self,
        opportunity: Dict,
        probability: float,
        confidence: float
    ) -> float:
        """Calculate risk score for the opportunity"""
        risk = 0.0

        # Uncertainty risk
        risk += (1 - confidence) * 0.3

        # Probability risk (extreme predictions are riskier)
        prob_risk = 1 - abs(probability - 0.5) * 2
        risk += prob_risk * 0.2

        # Volume risk
        buy_vol = opportunity.get("buy_volume", 0)
        sell_vol = opportunity.get("sell_volume", 0)
        quantity = opportunity.get("quantity", min(buy_vol, sell_vol))

        if min(buy_vol, sell_vol) > 0:
            size_ratio = quantity / min(buy_vol, sell_vol)
            risk += min(0.2, size_ratio * 0.1)

        # Toxicity risk
        toxicity = opportunity.get("toxicity_score", 0)
        risk += toxicity * 0.2

        # Spread risk
        spread_percentile = opportunity.get("spread_percentile", 0.5)
        if spread_percentile > 0.7:
            risk += 0.1

        return min(1.0, risk)

    def record_outcome(
        self,
        opportunity_id: str,
        was_profitable: bool,
        actual_profit: float,
        predicted_probability: float
    ):
        """Record actual outcome for drift detection"""
        self.drift_detector.update(predicted_probability, 1 if was_profitable else 0)

    def save(self, path: str):
        """Save model to disk"""
        if not self._is_trained:
            logger.warning("Model not trained, nothing to save")
            return

        save_dict = {
            'model': self.model,
            'calibrator': self.calibrator,
            'version': self.model_version,
            'feature_names': self.feature_pipeline.feature_names,
            'venue_pair_profit': self.feature_pipeline._venue_pair_profit,
            'hour_profit': self.feature_pipeline._hour_profit,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        try:
            with open(path, 'rb') as f:
                save_dict = pickle.load(f)

            self.model = save_dict['model']
            self.calibrator = save_dict.get('calibrator')
            self.model_version = save_dict.get('version', '1.0')
            self.feature_pipeline.feature_names = save_dict.get('feature_names', [])
            self.feature_pipeline._venue_pair_profit = save_dict.get('venue_pair_profit', {})
            self.feature_pipeline._hour_profit = save_dict.get('hour_profit', {})
            self._is_trained = True

            logger.info(f"Model loaded from {path}, version: {self.model_version}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def get_stats(self) -> Dict:
        """Get predictor statistics"""
        return {
            "is_trained": self._is_trained,
            "model_version": self.model_version,
            "training_samples": len(self.training_buffer),
            "samples_since_retrain": self.samples_since_retrain,
            "predictions_made": self.predictions_made,
            "avg_prediction_time_ms": (
                self.total_prediction_time_ms / self.predictions_made
                if self.predictions_made > 0 else 0
            ),
            "drift_detected": self.drift_detector.drift_detected,
            "current_error": self.drift_detector.current_error,
            "lightgbm_available": LIGHTGBM_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "unstable_features": self.importance_tracker.get_unstable_features()
        }


# Test function
def test_production_ml():
    """Test the production ML predictor"""
    print("Testing Production ML Predictor")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
    print(f"Scikit-learn available: {SKLEARN_AVAILABLE}")
    print()

    predictor = ProductionMLPredictor(min_training_samples=50)

    # Generate synthetic training data
    print("Generating training data...")
    np.random.seed(42)

    for i in range(200):
        spread = np.random.uniform(0.1, 0.5)
        volume = np.random.uniform(100, 1000)
        toxicity = np.random.uniform(0, 0.5)

        opportunity = {
            "id": f"train_{i}",
            "buy_price": 100,
            "sell_price": 100 + spread,
            "buy_volume": volume,
            "sell_volume": volume * np.random.uniform(0.8, 1.2),
            "buy_venue": np.random.choice(["kraken", "coinbase"]),
            "sell_venue": np.random.choice(["polymarket", "kraken"]),
            "market": "BTC",
            "obi": np.random.uniform(-0.3, 0.3),
            "toxicity_score": toxicity,
            "depth_ratio": np.random.uniform(0.5, 2),
            "estimated_fees": spread * 0.1,
            "estimated_slippage": spread * 0.05,
            "gross_profit": spread * volume
        }

        # Simulate profitability based on features
        profit_prob = 0.5 + 0.3 * spread - 0.5 * toxicity + 0.1 * (volume / 1000)
        was_profitable = np.random.random() < profit_prob
        actual_profit = spread * volume * (1 if was_profitable else -0.5)

        predictor.add_training_example(opportunity, was_profitable, actual_profit)

    # Test prediction
    print("\nTesting prediction...")
    test_opp = {
        "id": "test_1",
        "buy_price": 100,
        "sell_price": 100.3,
        "buy_volume": 500,
        "sell_volume": 450,
        "buy_venue": "kraken",
        "sell_venue": "polymarket",
        "market": "BTC",
        "obi": 0.1,
        "toxicity_score": 0.2,
        "depth_ratio": 1.2,
        "estimated_fees": 0.03,
        "estimated_slippage": 0.015,
        "gross_profit": 150
    }

    result = predictor.predict(test_opp)

    print(f"\nPrediction Result:")
    print(f"  Probability: {result.probability:.2%}")
    print(f"  Raw Probability: {result.raw_probability:.2%}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Expected Profit: ${result.expected_profit:.2f}")
    print(f"  Risk Score: {result.risk_score:.2f}")
    print(f"  Should Execute: {result.should_execute}")
    print(f"  Prediction Time: {result.prediction_time_ms:.3f}ms")

    print(f"\nTop Feature Contributions:")
    for name, contrib in sorted(
        result.feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]:
        print(f"  {name}: {contrib:.4f}")

    print(f"\nModel Stats:")
    stats = predictor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_production_ml()
