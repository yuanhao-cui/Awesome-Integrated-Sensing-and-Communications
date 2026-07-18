"""
Fairness Metrics for ISAC Resource Allocation.

Implements max-min fairness and proportional fairness metrics from
"Sensing as a Service in 6G Perceptive Networks".
"""

import numpy as np
from typing import Optional
from enum import Enum


class FairnessType(Enum):
    """Fairness types for resource allocation."""
    MAXMIN = "maxmin"           # Max-min fairness
    PROPORTIONAL = "proportional"  # Proportional fairness
    SUM = "sum"                 # Sum rate/comprehensiveness
    WEIGHTED_SUM = "weighted"   # Weighted sum


class FairnessMetrics:
    """
    Fairness metrics for ISAC resource allocation.

    Implements various fairness criteria used in the optimization framework.
    """

    def __init__(self, weights: Optional[np.ndarray] = None):
        """
        Initialize Fairness Metrics.

        Parameters
        ----------
        weights : np.ndarray, optional
            Weights for weighted sum fairness
        """
        self.weights = weights

    def compute_maxmin(self, values: np.ndarray) -> float:
        """
        Compute max-min fairness objective.

        max min_i x_i

        Parameters
        ----------
        values : np.ndarray
            Values to optimize (N,)

        Returns
        -------
        objective : float
            Minimum value across all entities
        """
        return np.min(values)

    def compute_sum(self, values: np.ndarray) -> float:
        """
        Compute sum objective (comprehensiveness).

        max Σ_i x_i

        Parameters
        ----------
        values : np.ndarray
            Values to optimize (N,)

        Returns
        -------
        objective : float
            Sum of all values
        """
        return np.sum(values)

    def compute_proportional_fairness(self, values: np.ndarray) -> float:
        """
        Compute proportional fairness objective.

        max Σ_i log(x_i)

        This ensures proportional fairness among all entities.

        Parameters
        ----------
        values : np.ndarray
            Values to optimize (N,)

        Returns
        -------
        objective : float
            Proportional fairness metric
        """
        # Add small epsilon to avoid log(0)
        return np.sum(np.log(values + 1e-10))

    def compute_weighted_sum(self, values: np.ndarray,
                              weights: Optional[np.ndarray] = None) -> float:
        """
        Compute weighted sum objective.

        max Σ_i w_i * x_i

        Parameters
        ----------
        values : np.ndarray
            Values to optimize (N,)
        weights : np.ndarray, optional
            Weights for each value. If None, uses stored weights.

        Returns
        -------
        objective : float
            Weighted sum
        """
        if weights is None:
            weights = self.weights

        if weights is None:
            weights = np.ones_like(values)

        return np.sum(weights * values)

    def compute_jain_fairness_index(self, values: np.ndarray) -> float:
        """
        Compute Jain's Fairness Index.

        J = (Σ_i x_i)² / (N * Σ_i x_i²)

        Range: [1/N, 1] where 1 is perfectly fair.

        Parameters
        ----------
        values : np.ndarray
            Values to evaluate (N,)

        Returns
        -------
        jfi : float
            Jain's Fairness Index
        """
        n = len(values)
        if n == 0:
            return 1.0

        sum_x = np.sum(values)
        sum_x_sq = np.sum(values**2)

        if sum_x_sq == 0:
            return 1.0

        return (sum_x**2) / (n * sum_x_sq)

    def compute_min_max_ratio(self, values: np.ndarray) -> float:
        """
        Compute min/max ratio (another fairness measure).

        Ratio = min(x_i) / max(x_i)

        Parameters
        ----------
        values : np.ndarray
            Values to evaluate (N,)

        Returns
        -------
        ratio : float
            Min/max ratio [0, 1]
        """
        if len(values) == 0:
            return 1.0

        min_val = np.min(values)
        max_val = np.max(values)

        if max_val == 0:
            return 1.0

        return min_val / max_val

    def compute_gini_coefficient(self, values: np.ndarray) -> float:
        """
        Compute Gini coefficient (inequality measure).

        G = (Σ_i Σ_j |x_i - x_j|) / (2N Σ_i x_i)

        Range: [0, 1] where 0 is perfectly equal.

        Parameters
        ----------
        values : np.ndarray
            Values to evaluate (N,)

        Returns
        -------
        gini : float
            Gini coefficient
        """
        n = len(values)
        if n <= 1:
            return 0.0

        sorted_vals = np.sort(values)
        index = np.arange(1, n + 1)

        return (2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / \
               (n * np.sum(sorted_vals))

    def compute_objective(self, values: np.ndarray,
                          fairness_type: FairnessType,
                          weights: Optional[np.ndarray] = None) -> float:
        """
        Compute fairness objective based on specified type.

        Parameters
        ----------
        values : np.ndarray
            Values to optimize (N,)
        fairness_type : FairnessType
            Type of fairness metric
        weights : np.ndarray, optional
            Weights for weighted sum

        Returns
        -------
        objective : float
            Computed objective value
        """
        if fairness_type == FairnessType.MAXMIN:
            return self.compute_maxmin(values)
        elif fairness_type == FairnessType.PROPORTIONAL:
            return self.compute_proportional_fairness(values)
        elif fairness_type == FairnessType.SUM:
            return self.compute_sum(values)
        elif fairness_type == FairnessType.WEIGHTED_SUM:
            return self.compute_weighted_sum(values, weights)
        else:
            raise ValueError(f"Unknown fairness type: {fairness_type}")

    def compute_gradient_maxmin(self, values: np.ndarray) -> np.ndarray:
        """
        Compute gradient of max-min objective.

        ∂min(x)/∂x_i = 1 if i = argmin(x), 0 otherwise

        Parameters
        ----------
        values : np.ndarray
            Values (N,)

        Returns
        -------
        gradient : np.ndarray
            Gradient (N,)
        """
        gradient = np.zeros_like(values)
        min_idx = np.argmin(values)
        gradient[min_idx] = 1.0
        return gradient

    def compute_gradient_proportional(self, values: np.ndarray) -> np.ndarray:
        """
        Compute gradient of proportional fairness objective.

        ∂Σlog(x)/∂x_i = 1/x_i

        Parameters
        ----------
        values : np.ndarray
            Values (N,)

        Returns
        -------
        gradient : np.ndarray
            Gradient (N,)
        """
        return 1.0 / (values + 1e-10)

    def compute_gradient(self, values: np.ndarray,
                         fairness_type: FairnessType) -> np.ndarray:
        """
        Compute gradient of fairness objective.

        Parameters
        ----------
        values : np.ndarray
            Values (N,)
        fairness_type : FairnessType
            Type of fairness metric

        Returns
        -------
        gradient : np.ndarray
            Gradient (N,)
        """
        if fairness_type == FairnessType.MAXMIN:
            return self.compute_gradient_maxmin(values)
        elif fairness_type == FairnessType.PROPORTIONAL:
            return self.compute_gradient_proportional(values)
        elif fairness_type == FairnessType.SUM:
            return np.ones_like(values)
        elif fairness_type == FairnessType.WEIGHTED_SUM:
            return self.weights if self.weights is not None else np.ones_like(values)
        else:
            raise ValueError(f"Unknown fairness type: {fairness_type}")

    def evaluate_fairness_metrics(self, values: np.ndarray) -> dict:
        """
        Evaluate all fairness metrics for given values.

        Parameters
        ----------
        values : np.ndarray
            Values to evaluate (N,)

        Returns
        -------
        metrics : dict
            Dictionary of fairness metrics
        """
        return {
            'jain_fairness': self.compute_jain_fairness_index(values),
            'min_max_ratio': self.compute_min_max_ratio(values),
            'gini_coefficient': self.compute_gini_coefficient(values),
            'min_value': np.min(values),
            'max_value': np.max(values),
            'mean_value': np.mean(values),
            'std_value': np.std(values)
        }
