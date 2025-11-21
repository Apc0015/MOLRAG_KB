"""
Evaluation Module - Phase 6 Complete

Includes:
- Metrics calculation (ROC-AUC, AUPR, Recall@K, etc.)
- Dashboard for real-time monitoring
- Benchmark datasets (BACE, CYP450, HIV, Tox21, etc.)
- Expert validation pipeline (target: 75%+ approval)
- A/B testing framework for model comparison
"""

from .metrics import (
    RetrievalMetrics,
    PredictionMetrics,
    ExplanationMetrics,
    EvaluationSuite
)
from .dashboard import MetricsDashboard
from .benchmarks import BenchmarkLoader, BenchmarkDataset
from .expert_validation import ExpertValidationPipeline, ExpertReview
from .ab_testing import ABTestingFramework, ABTestResult, Variant

__all__ = [
    "RetrievalMetrics",
    "PredictionMetrics",
    "ExplanationMetrics",
    "EvaluationSuite",
    "MetricsDashboard",
    "BenchmarkLoader",
    "BenchmarkDataset",
    "ExpertValidationPipeline",
    "ExpertReview",
    "ABTestingFramework",
    "ABTestResult",
    "Variant"
]
