"""
Evaluation metrics for MolRAG

Based on blueprint specifications:
- Retrieval Quality: Recall@K, Precision@K, MRR, Avg Tanimoto
- Prediction Quality: ROC-AUC, AUPR, RMSE, MAE
- Explanation Quality: Path relevance, expert validation, citation accuracy
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score
)

from ..data.models import RetrievalResult, PredictionResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalMetrics:
    """Metrics for retrieval quality evaluation"""

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K

        Args:
            retrieved_ids: List of retrieved molecule IDs (ordered)
            relevant_ids: List of ground truth relevant IDs
            k: Top-K cutoff

        Returns:
            Recall@K score (0-1)
        """
        if not relevant_ids:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_retrieved = set(top_k) & set(relevant_ids)

        recall = len(relevant_retrieved) / len(relevant_ids)
        return recall

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K

        Args:
            retrieved_ids: List of retrieved molecule IDs
            relevant_ids: List of ground truth relevant IDs
            k: Top-K cutoff

        Returns:
            Precision@K score (0-1)
        """
        if not retrieved_ids:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_retrieved = set(top_k) & set(relevant_ids)

        precision = len(relevant_retrieved) / k
        return precision

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ids_list: List[List[str]],
        relevant_ids_list: List[List[str]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)

        Args:
            retrieved_ids_list: List of retrieved ID lists (one per query)
            relevant_ids_list: List of relevant ID lists

        Returns:
            MRR score (0-1)
        """
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
            # Find rank of first relevant item
            rank = None
            for i, ret_id in enumerate(retrieved, 1):
                if ret_id in relevant:
                    rank = i
                    break

            if rank:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

    @staticmethod
    def average_tanimoto(
        retrieval_results: List[RetrievalResult]
    ) -> float:
        """
        Calculate average Tanimoto similarity

        Args:
            retrieval_results: List of retrieval results

        Returns:
            Average Tanimoto score
        """
        tanimoto_scores = [
            r.tanimoto_score for r in retrieval_results
            if r.tanimoto_score is not None
        ]

        if not tanimoto_scores:
            return 0.0

        return np.mean(tanimoto_scores)


class PredictionMetrics:
    """Metrics for property prediction evaluation"""

    @staticmethod
    def roc_auc(
        y_true: List[int],
        y_pred_proba: List[float]
    ) -> float:
        """
        Calculate ROC-AUC for binary classification

        Args:
            y_true: Ground truth labels (0/1)
            y_pred_proba: Predicted probabilities

        Returns:
            ROC-AUC score (0-1)
        """
        try:
            return roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            logger.error(f"ROC-AUC calculation failed: {e}")
            return 0.0

    @staticmethod
    def aupr(
        y_true: List[int],
        y_pred_proba: List[float]
    ) -> float:
        """
        Calculate Area Under Precision-Recall curve (AUPR)

        Useful for imbalanced datasets

        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities

        Returns:
            AUPR score (0-1)
        """
        try:
            return average_precision_score(y_true, y_pred_proba)
        except Exception as e:
            logger.error(f"AUPR calculation failed: {e}")
            return 0.0

    @staticmethod
    def rmse(
        y_true: List[float],
        y_pred: List[float]
    ) -> float:
        """
        Calculate Root Mean Squared Error (for regression)

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            RMSE
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(
        y_true: List[float],
        y_pred: List[float]
    ) -> float:
        """
        Calculate Mean Absolute Error (for regression)

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            MAE
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def accuracy(
        y_true: List[int],
        y_pred: List[int]
    ) -> float:
        """Calculate accuracy"""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def f1(
        y_true: List[int],
        y_pred: List[int]
    ) -> float:
        """Calculate F1 score"""
        return f1_score(y_true, y_pred, average='binary')


class ExplanationMetrics:
    """Metrics for explanation quality evaluation"""

    @staticmethod
    def path_relevance_score(
        pathway: List[str],
        hops: int,
        path_count: int
    ) -> float:
        """
        Calculate pathway relevance score

        Based on blueprint formula

        Args:
            pathway: List of relationship types
            hops: Number of hops
            path_count: Number of paths

        Returns:
            Relevance score (0-1)
        """
        # Base edge confidence
        edge_confidence = 0.8
        base_score = edge_confidence * 0.6

        # Hop penalty
        hop_penalty = 0.2 * hops
        hop_score = max(0, 1.0 - hop_penalty)

        # Relationship boost
        high_value_rels = ['BINDS', 'TARGETS', 'TREATS']
        rel_boost = sum(0.1 for rel in pathway if rel in high_value_rels)

        # Frequency boost
        import math
        freq_boost = min(0.2, math.log(1 + path_count) * 0.05)

        score = base_score * hop_score + rel_boost + freq_boost
        return min(1.0, score)


class EvaluationSuite:
    """Complete evaluation suite for MolRAG"""

    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.prediction_metrics = PredictionMetrics()
        self.explanation_metrics = ExplanationMetrics()

    def evaluate_retrieval(
        self,
        retrieval_results: List[List[RetrievalResult]],
        ground_truth: List[List[str]],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality

        Args:
            retrieval_results: List of retrieval results per query
            ground_truth: List of relevant molecule IDs per query
            k_values: K values for Recall@K and Precision@K

        Returns:
            Dictionary of metric scores
        """
        results = {}

        # Extract IDs
        retrieved_ids_list = [
            [r.smiles for r in results]
            for results in retrieval_results
        ]

        # Recall@K and Precision@K
        for k in k_values:
            recall_scores = [
                self.retrieval_metrics.recall_at_k(retrieved, relevant, k)
                for retrieved, relevant in zip(retrieved_ids_list, ground_truth)
            ]
            results[f'recall@{k}'] = np.mean(recall_scores)

            precision_scores = [
                self.retrieval_metrics.precision_at_k(retrieved, relevant, k)
                for retrieved, relevant in zip(retrieved_ids_list, ground_truth)
            ]
            results[f'precision@{k}'] = np.mean(precision_scores)

        # MRR
        results['mrr'] = self.retrieval_metrics.mean_reciprocal_rank(
            retrieved_ids_list, ground_truth
        )

        # Average Tanimoto
        all_results = [r for results in retrieval_results for r in results]
        results['avg_tanimoto'] = self.retrieval_metrics.average_tanimoto(
            all_results
        )

        return results

    def evaluate_prediction(
        self,
        predictions: List[PredictionResult],
        ground_truth: List[Any],
        task_type: str = 'classification'
    ) -> Dict[str, float]:
        """
        Evaluate prediction quality

        Args:
            predictions: List of prediction results
            ground_truth: Ground truth values
            task_type: 'classification' or 'regression'

        Returns:
            Dictionary of metric scores
        """
        results = {}

        # Extract predictions and confidences
        y_pred = []
        y_conf = []

        for pred in predictions:
            # Convert prediction to binary/numerical
            if task_type == 'classification':
                if pred.prediction.lower() in ['yes', '1', 'true']:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            else:
                try:
                    y_pred.append(float(pred.prediction))
                except:
                    y_pred.append(0.0)

            y_conf.append(pred.confidence)

        # Calculate metrics
        if task_type == 'classification':
            results['roc_auc'] = self.prediction_metrics.roc_auc(
                ground_truth, y_conf
            )
            results['aupr'] = self.prediction_metrics.aupr(
                ground_truth, y_conf
            )
            results['accuracy'] = self.prediction_metrics.accuracy(
                ground_truth, y_pred
            )
            results['f1'] = self.prediction_metrics.f1(
                ground_truth, y_pred
            )
        else:
            results['rmse'] = self.prediction_metrics.rmse(
                ground_truth, y_pred
            )
            results['mae'] = self.prediction_metrics.mae(
                ground_truth, y_pred
            )

        return results

    def generate_report(
        self,
        retrieval_metrics: Dict[str, float],
        prediction_metrics: Dict[str, float]
    ) -> str:
        """
        Generate evaluation report

        Args:
            retrieval_metrics: Retrieval metric scores
            prediction_metrics: Prediction metric scores

        Returns:
            Formatted report string
        """
        report = "=" * 60 + "\n"
        report += "MolRAG Evaluation Report\n"
        report += "=" * 60 + "\n\n"

        report += "RETRIEVAL QUALITY:\n"
        report += "-" * 40 + "\n"
        for metric, value in retrieval_metrics.items():
            report += f"{metric:20s}: {value:.4f}\n"

        report += "\nPREDICTION QUALITY:\n"
        report += "-" * 40 + "\n"
        for metric, value in prediction_metrics.items():
            report += f"{metric:20s}: {value:.4f}\n"

        report += "\n" + "=" * 60 + "\n"

        return report
