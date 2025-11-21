"""
Phase 6: Metrics Dashboard
Real-time visualization of system performance
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import pandas as pd
from pathlib import Path

from .metrics import (
    calculate_roc_auc,
    calculate_aupr,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_tanimoto_similarity
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetricsDashboard:
    """
    Production Metrics Dashboard

    Visualizes:
    - Retrieval quality (Recall@K, Precision@K, MRR)
    - Prediction quality (ROC-AUC, AUPR, RMSE)
    - Explanation quality (Path relevance, expert approval)
    - System performance (latency, throughput)
    """

    def __init__(self, results_dir: Path = None):
        """
        Initialize dashboard

        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = results_dir or Path("evaluation_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_retrieval_dashboard(self, retrieval_results: Dict):
        """
        Create retrieval quality dashboard

        Args:
            retrieval_results: Dict with retrieval metrics
        """
        st.header("🔍 Retrieval Quality Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Recall@10",
                f"{retrieval_results.get('recall_at_10', 0):.3f}",
                delta=f"+{retrieval_results.get('recall_at_10_delta', 0):.3f}"
            )

        with col2:
            st.metric(
                "MRR",
                f"{retrieval_results.get('mrr', 0):.3f}",
                delta=f"+{retrieval_results.get('mrr_delta', 0):.3f}"
            )

        with col3:
            st.metric(
                "Avg Tanimoto",
                f"{retrieval_results.get('avg_tanimoto', 0):.3f}",
                delta=f"+{retrieval_results.get('avg_tanimoto_delta', 0):.3f}"
            )

        # Recall@K curve
        if 'recall_at_k' in retrieval_results:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(retrieval_results['recall_at_k'].keys()),
                y=list(retrieval_results['recall_at_k'].values()),
                mode='lines+markers',
                name='Recall@K'
            ))
            fig.update_layout(
                title="Recall@K Performance",
                xaxis_title="K",
                yaxis_title="Recall",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)

    def create_prediction_dashboard(self, prediction_results: Dict):
        """
        Create prediction quality dashboard

        Args:
            prediction_results: Dict with prediction metrics
        """
        st.header("🎯 Prediction Quality Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "ROC-AUC",
                f"{prediction_results.get('roc_auc', 0):.3f}",
                delta=f"+{prediction_results.get('roc_auc_delta', 0):.3f}"
            )

        with col2:
            st.metric(
                "AUPR",
                f"{prediction_results.get('aupr', 0):.3f}",
                delta=f"+{prediction_results.get('aupr_delta', 0):.3f}"
            )

        with col3:
            st.metric(
                "Accuracy",
                f"{prediction_results.get('accuracy', 0):.3f}",
                delta=f"+{prediction_results.get('accuracy_delta', 0):.3f}"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{prediction_results.get('f1', 0):.3f}",
                delta=f"+{prediction_results.get('f1_delta', 0):.3f}"
            )

        # ROC Curve
        if 'fpr' in prediction_results and 'tpr' in prediction_results:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prediction_results['fpr'],
                y=prediction_results['tpr'],
                mode='lines',
                name=f'ROC (AUC = {prediction_results.get("roc_auc", 0):.3f})',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
            st.plotly_chart(fig, use_container_width=True)

    def create_explanation_dashboard(self, explanation_results: Dict):
        """
        Create explanation quality dashboard

        Args:
            explanation_results: Dict with explanation metrics
        """
        st.header("💡 Explanation Quality Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Path Relevance",
                f"{explanation_results.get('path_relevance', 0):.3f}"
            )

        with col2:
            st.metric(
                "Expert Approval",
                f"{explanation_results.get('expert_approval', 0):.1%}"
            )

        with col3:
            st.metric(
                "Citation Accuracy",
                f"{explanation_results.get('citation_accuracy', 0):.1%}"
            )

    def create_performance_dashboard(self, performance_results: Dict):
        """
        Create system performance dashboard

        Args:
            performance_results: Dict with performance metrics
        """
        st.header("⚡ System Performance Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Avg Latency",
                f"{performance_results.get('avg_latency_ms', 0):.0f}ms"
            )

        with col2:
            st.metric(
                "Throughput",
                f"{performance_results.get('throughput_qps', 0):.1f} q/s"
            )

        with col3:
            st.metric(
                "Success Rate",
                f"{performance_results.get('success_rate', 0):.1%}"
            )

        # Latency distribution
        if 'latency_distribution' in performance_results:
            fig = px.histogram(
                performance_results['latency_distribution'],
                nbins=50,
                title="Latency Distribution"
            )
            fig.update_xaxes(title="Latency (ms)")
            fig.update_yaxes(title="Count")
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Run the dashboard"""
        st.set_page_config(page_title="MolRAG Metrics Dashboard", layout="wide")

        st.title("🧬 MolRAG Production Metrics Dashboard")

        st.markdown("""
        Real-time monitoring of MolRAG system performance across all metrics.
        """)

        # Load latest results
        try:
            retrieval_results = self.load_results("retrieval")
            prediction_results = self.load_results("prediction")
            explanation_results = self.load_results("explanation")
            performance_results = self.load_results("performance")

            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "Retrieval", "Prediction", "Explanation", "Performance"
            ])

            with tab1:
                self.create_retrieval_dashboard(retrieval_results)

            with tab2:
                self.create_prediction_dashboard(prediction_results)

            with tab3:
                self.create_explanation_dashboard(explanation_results)

            with tab4:
                self.create_performance_dashboard(performance_results)

        except Exception as e:
            st.error(f"Error loading results: {e}")
            st.info("Run evaluations first: python scripts/evaluate.py")

    def load_results(self, result_type: str) -> Dict:
        """Load evaluation results"""
        results_file = self.results_dir / f"{result_type}_latest.json"
        if results_file.exists():
            import json
            with open(results_file, 'r') as f:
                return json.load(f)
        return {}


if __name__ == "__main__":
    dashboard = MetricsDashboard()
    dashboard.run()
