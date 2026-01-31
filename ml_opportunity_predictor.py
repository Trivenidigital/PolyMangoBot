"""
Machine Learning Opportunity Predictor
Uses ML to predict when arbitrage opportunities will appear
With proper cross-validation and statistical foundations
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import statistics
import math
import random
import logging
from datetime import datetime

logger = logging.getLogger("PolyMangoBot.ml")


@dataclass
class TrainingFeatures:
    """Features for ML training with validation"""
    spread_percent: float
    bid_volume: float
    ask_volume: float
    spread_volatility: float
    order_flow_imbalance: float  # (buy_orders - sell_orders) / total
    recent_price_trend: float  # -1 to 1, price direction
    time_of_day_hour: int
    volatility_5min: float
    bid_ask_ratio: float  # bid_qty / ask_qty
    label: int  # 1 if spread widened in next 5s, 0 otherwise
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self):
        """Validate feature values"""
        if not 0 <= self.time_of_day_hour < 24:
            raise ValueError(f"Invalid hour: {self.time_of_day_hour}")
        if not -1 <= self.recent_price_trend <= 1:
            self.recent_price_trend = max(-1, min(1, self.recent_price_trend))
        if self.label not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {self.label}")

    def to_array(self) -> List[float]:
        """Convert features to array for model input"""
        return [
            self.spread_percent,
            self.bid_volume,
            self.ask_volume,
            self.spread_volatility,
            self.order_flow_imbalance,
            self.recent_price_trend,
            float(self.time_of_day_hour),
            self.volatility_5min,
            self.bid_ask_ratio,
        ]


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: Dict[str, int]

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.3f} | "
            f"Precision: {self.precision:.3f} | "
            f"Recall: {self.recall:.3f} | "
            f"F1: {self.f1_score:.3f} | "
            f"AUC-ROC: {self.auc_roc:.3f}"
        )


class MLOpportunityPredictor:
    """
    Machine Learning model to predict arbitrage opportunities

    Features:
    - Online learning with sliding window
    - K-fold cross-validation for model evaluation
    - Feature normalization with running statistics
    - Regularization to prevent overfitting
    """

    FEATURE_NAMES = [
        'spread_percent', 'bid_volume', 'ask_volume',
        'spread_volatility', 'order_flow_imbalance',
        'recent_price_trend', 'time_of_day_hour',
        'volatility_5min', 'bid_ask_ratio'
    ]

    def __init__(self, max_training_samples: int = 10000, regularization: float = 0.01):
        """
        Initialize predictor with bounded training data

        Args:
            max_training_samples: Maximum training samples to keep (prevents memory leak)
            regularization: L2 regularization strength
        """
        self.training_data: deque = deque(maxlen=max_training_samples)
        self.model_weights: Dict[str, float] = {}
        self.feature_stats: Dict[str, Tuple[float, float]] = {}  # feature -> (mean, std)
        self.is_trained = False
        self.regularization = regularization

        # Running statistics for online normalization
        self._running_means: Dict[str, float] = {f: 0.0 for f in self.FEATURE_NAMES}
        self._running_vars: Dict[str, float] = {f: 1.0 for f in self.FEATURE_NAMES}
        self._sample_count = 0

        # Model performance tracking
        self.last_metrics: Optional[ModelMetrics] = None
        self.validation_history: List[float] = []

    def add_training_data(self, features: TrainingFeatures):
        """Add labeled example to training set with online statistics update"""
        self.training_data.append(features)
        self._update_running_stats(features)

    def _update_running_stats(self, features: TrainingFeatures):
        """Update running mean and variance (Welford's algorithm)"""
        self._sample_count += 1
        n = self._sample_count

        for fname in self.FEATURE_NAMES:
            value = getattr(features, fname)
            old_mean = self._running_means[fname]

            # Update mean
            new_mean = old_mean + (value - old_mean) / n
            self._running_means[fname] = new_mean

            # Update variance (for n > 1)
            if n > 1:
                old_var = self._running_vars[fname]
                new_var = old_var + ((value - old_mean) * (value - new_mean) - old_var) / n
                self._running_vars[fname] = max(new_var, 1e-8)  # Prevent zero variance

    def normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize feature using running statistics"""
        if feature_name not in self._running_means:
            return 0

        mean = self._running_means[feature_name]
        std = math.sqrt(self._running_vars[feature_name])

        if std < 1e-8:
            return 0

        # Z-score normalization
        normalized = (value - mean) / std

        # Clip to -3 to +3 range (within 3 standard deviations)
        return max(-3, min(3, normalized))

    def _calculate_feature_stats(self) -> None:
        """Calculate feature statistics from training data"""
        data_list = list(self.training_data)

        for fname in self.FEATURE_NAMES:
            values = [getattr(d, fname) for d in data_list]
            if len(values) > 1:
                mean = statistics.mean(values)
                stdev = statistics.stdev(values)
                self.feature_stats[fname] = (mean, max(stdev, 1e-8))
            else:
                self.feature_stats[fname] = (0, 1)

    def _k_fold_split(self, data: List, k: int = 5) -> List[Tuple[List, List]]:
        """Split data into k folds for cross-validation"""
        shuffled = data.copy()
        random.shuffle(shuffled)

        fold_size = len(shuffled) // k
        folds = []

        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(shuffled)

            test_fold = shuffled[start:end]
            train_fold = shuffled[:start] + shuffled[end:]
            folds.append((train_fold, test_fold))

        return folds

    def train(self, k_folds: int = 5) -> bool:
        """
        Train the model with k-fold cross-validation

        Uses gradient descent with L2 regularization for better generalization
        """
        data_list = list(self.training_data)

        if len(data_list) < 20:
            logger.warning(f"Not enough training data: {len(data_list)} samples")
            return False

        logger.info(f"Training model on {len(data_list)} samples with {k_folds}-fold CV")

        # Calculate feature statistics
        self._calculate_feature_stats()

        # Cross-validation
        cv_scores = []
        all_weights = []

        for fold_idx, (train_data, test_data) in enumerate(self._k_fold_split(data_list, k_folds)):
            # Train on this fold
            weights = self._train_fold(train_data)
            all_weights.append(weights)

            # Evaluate on test fold
            accuracy = self._evaluate_fold(weights, test_data)
            cv_scores.append(accuracy)
            logger.debug(f"Fold {fold_idx + 1}: accuracy = {accuracy:.3f}")

        # Average weights across folds
        self.model_weights = {}
        for fname in self.FEATURE_NAMES:
            self.model_weights[fname] = statistics.mean(
                w.get(fname, 0) for w in all_weights
            )

        # Store validation results
        mean_cv_score = statistics.mean(cv_scores)
        self.validation_history.append(mean_cv_score)

        # Calculate final metrics
        self.last_metrics = self._calculate_metrics(data_list)

        self.is_trained = True
        logger.info(
            f"Model trained: CV accuracy = {mean_cv_score:.3f}, "
            f"Metrics: {self.last_metrics}"
        )
        return True

    def _train_fold(self, train_data: List[TrainingFeatures]) -> Dict[str, float]:
        """Train on a single fold using gradient descent"""
        weights = {fname: 0.0 for fname in self.FEATURE_NAMES}
        learning_rate = 0.01
        epochs = 100

        for _ in range(epochs):
            for sample in train_data:
                # Forward pass
                score = sum(
                    weights[fname] * self.normalize_feature(fname, getattr(sample, fname))
                    for fname in self.FEATURE_NAMES
                )
                prediction = 1.0 / (1.0 + math.exp(-score))  # Sigmoid

                # Backward pass (gradient descent)
                error = sample.label - prediction
                for fname in self.FEATURE_NAMES:
                    normalized_value = self.normalize_feature(fname, getattr(sample, fname))
                    # Gradient with L2 regularization
                    gradient = error * normalized_value - self.regularization * weights[fname]
                    weights[fname] += learning_rate * gradient

        return weights

    def _evaluate_fold(self, weights: Dict[str, float], test_data: List[TrainingFeatures]) -> float:
        """Evaluate accuracy on test fold"""
        correct = 0

        for sample in test_data:
            score = sum(
                weights[fname] * self.normalize_feature(fname, getattr(sample, fname))
                for fname in self.FEATURE_NAMES
            )
            prediction = 1 if 1.0 / (1.0 + math.exp(-score)) > 0.5 else 0
            if prediction == sample.label:
                correct += 1

        return correct / len(test_data) if test_data else 0

    def _calculate_metrics(self, data: List[TrainingFeatures]) -> ModelMetrics:
        """Calculate comprehensive model metrics"""
        tp = fp = tn = fn = 0

        for sample in data:
            pred = self.predict_opportunity(sample)
            predicted = 1 if pred['probability'] > 0.5 else 0
            actual = sample.label

            if predicted == 1 and actual == 1:
                tp += 1
            elif predicted == 1 and actual == 0:
                fp += 1
            elif predicted == 0 and actual == 0:
                tn += 1
            else:
                fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Simplified AUC-ROC estimation
        auc_roc = (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0.5

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        )

    def predict_opportunity(self, features: TrainingFeatures) -> Dict:
        """
        Predict if an opportunity will appear

        Returns:
            {
                'will_widen': bool,
                'probability': float (0-1),
                'confidence': float (0-1),
                'feature_importance': {feature -> contribution}
            }
        """

        if not self.is_trained:
            return {'will_widen': False, 'probability': 0.5, 'confidence': 0}

        # Calculate model output (weighted sum of normalized features)
        feature_dict = {
            'spread_percent': features.spread_percent,
            'bid_volume': features.bid_volume,
            'ask_volume': features.ask_volume,
            'spread_volatility': features.spread_volatility,
            'order_flow_imbalance': features.order_flow_imbalance,
            'recent_price_trend': features.recent_price_trend,
            'volatility_5min': features.volatility_5min,
            'bid_ask_ratio': features.bid_ask_ratio,
        }

        score = 0.0
        contributions = {}

        for fname, value in feature_dict.items():
            normalized = self.normalize_feature(fname, value)
            weight = self.model_weights.get(fname, 0)

            contribution = normalized * weight
            score += contribution
            contributions[fname] = contribution

        # Convert raw score to probability using sigmoid
        probability = 1.0 / (1.0 + (2.718 ** (-score)))  # Sigmoid

        # Confidence based on feature agreement
        abs_contributions = [abs(c) for c in contributions.values()]
        confidence = sum(abs_contributions) / (len(abs_contributions) + 1)

        return {
            'will_widen': probability > 0.5,
            'probability': probability,
            'confidence': min(confidence, 1.0),
            'feature_importance': contributions
        }

    def get_most_important_features(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """Get features most predictive of opportunities"""

        if not self.model_weights:
            return []

        sorted_features = sorted(
            self.model_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return sorted_features[:top_n]


class EnsemblePredictor:
    """
    Combines multiple prediction models for robustness

    Uses:
    - ML model
    - Statistical signals
    - Heuristic rules
    """

    def __init__(self):
        self.ml_model = MLOpportunityPredictor()

    def predict_spread_expansion(self, features: TrainingFeatures,
                                signal_data: Dict) -> Dict:
        """
        Make ensemble prediction about spread expansion

        Combines:
        1. ML model prediction
        2. Statistical mean reversion
        3. Momentum signals
        4. Liquidity signals
        """

        ml_pred = self.ml_model.predict_opportunity(features)

        # Statistical signal: if spread is low, mean reversion suggests it will widen
        spread_percentile = signal_data.get('spread_percentile', 0.5)  # 0-1
        mean_reversion_signal = 0.5 if spread_percentile < 0.3 else -0.5 if spread_percentile > 0.7 else 0

        # Momentum signal: order flow imbalance predicts continuation
        momentum_signal = features.order_flow_imbalance * 0.5

        # Liquidity signal: thin books predict wider spreads
        bid_ask_ratio = features.bid_ask_ratio
        liquidity_signal = -0.3 if bid_ask_ratio < 0.5 or bid_ask_ratio > 2.0 else 0.1

        # Ensemble combination
        ensemble_score = (
            ml_pred['probability'] * 0.4 +  # ML prediction: 40%
            (0.5 + mean_reversion_signal) * 0.3 +  # Mean reversion: 30%
            (0.5 + momentum_signal) * 0.2 +  # Momentum: 20%
            (0.5 + liquidity_signal) * 0.1  # Liquidity: 10%
        )

        confidence = (ml_pred['confidence'] + 0.5) / 2  # Average confidences

        return {
            'will_expand': ensemble_score > 0.5,
            'probability': ensemble_score,
            'confidence': confidence,
            'component_scores': {
                'ml_model': ml_pred['probability'],
                'mean_reversion': 0.5 + mean_reversion_signal,
                'momentum': 0.5 + momentum_signal,
                'liquidity': 0.5 + liquidity_signal,
            }
        }


class OpportunitySignalGenerator:
    """Generates trading signals based on multiple indicators"""

    @staticmethod
    def generate_signal(
        prediction: Dict,
        current_spread: float,
        min_spread_threshold: float
    ) -> Dict:
        """
        Convert prediction into actionable trading signal

        Returns:
            {
                'signal_strength': float (-1 to +1),
                'action': 'buy_dip' | 'sell_rally' | 'wait' | 'none',
                'urgency': float (0-1),  # How quickly to act
            }
        """

        if not prediction.get('will_expand'):
            return {'signal_strength': 0, 'action': 'none', 'urgency': 0}

        # If spread is about to expand and is currently tight, it's a buy signal
        # (buy now before spread widens, limiting your fill slippage)

        if current_spread < min_spread_threshold:
            action = 'wait'  # Spread too tight, skip
            urgency = 0
        else:
            action = 'prepare'  # Get ready to trade
            urgency = prediction['confidence']

        signal_strength = prediction['probability'] - 0.5  # -0.5 to +0.5

        return {
            'signal_strength': signal_strength,
            'action': action,
            'urgency': urgency,
            'confidence': prediction['confidence']
        }


# Test
def test_ml_predictor():
    """Test ML predictor"""

    predictor = MLOpportunityPredictor()
    ensemble = EnsemblePredictor()

    # Generate synthetic training data
    print("Training model...")

    for i in range(50):
        spread = 0.5 + (i % 10) * 0.1
        bid_vol = 1000 + (i % 20) * 100
        ask_vol = 900 + (i % 20) * 100
        volatility = 0.3 + (i % 5) * 0.1

        features = TrainingFeatures(
            spread_percent=spread,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            spread_volatility=volatility,
            order_flow_imbalance=(bid_vol - ask_vol) / (bid_vol + ask_vol),
            recent_price_trend=-0.5 + (i % 10) / 10,
            time_of_day_hour=(i * 2) % 24,
            volatility_5min=volatility,
            bid_ask_ratio=bid_vol / ask_vol,
            label=1 if volatility > 0.5 else 0
        )

        predictor.add_training_data(features)

    predictor.train()

    # Test prediction
    test_features = TrainingFeatures(
        spread_percent=0.6,
        bid_volume=1200,
        ask_volume=950,
        spread_volatility=0.6,
        order_flow_imbalance=0.11,
        recent_price_trend=0.3,
        time_of_day_hour=14,
        volatility_5min=0.5,
        bid_ask_ratio=1.26,
        label=0  # Unknown
    )

    pred = predictor.predict_opportunity(test_features)
    print(f"Prediction: {pred}")

    # Ensemble prediction
    signal_data = {'spread_percentile': 0.4}
    ensemble_pred = ensemble.predict_spread_expansion(test_features, signal_data)
    print(f"Ensemble prediction: {ensemble_pred}")


if __name__ == "__main__":
    test_ml_predictor()
