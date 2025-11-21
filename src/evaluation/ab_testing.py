"""
Phase 6: A/B Testing Framework
Compare different CoT strategies, retrieval methods, and model configurations
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy import stats
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Variant:
    """A/B test variant"""
    name: str
    config: Dict
    results: List[float] = None
    sample_size: int = 0


@dataclass
class ABTestResult:
    """A/B test results"""
    test_name: str
    variant_a: Variant
    variant_b: Variant
    winner: Optional[str]
    p_value: float
    confidence_level: float
    effect_size: float
    is_significant: bool
    timestamp: datetime


class ABTestingFramework:
    """
    A/B Testing Framework for MolRAG

    Test comparisons:
    - CoT strategies (Struct-CoT vs Sim-CoT vs Path-CoT)
    - Retrieval methods (Vector vs Graph vs GNN vs Hybrid)
    - Model configurations (GPT-4 vs Claude vs Others)
    - Hyperparameters (top-k values, weights, etc.)
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize A/B testing framework

        Args:
            output_dir: Directory to store test results
        """
        self.output_dir = output_dir or Path("ab_tests")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_test(
        self,
        test_name: str,
        variant_a: Dict,
        variant_b: Dict,
        metric: str = "accuracy"
    ) -> str:
        """
        Create new A/B test

        Args:
            test_name: Name of the test
            variant_a: Configuration for variant A
            variant_b: Configuration for variant B
            metric: Metric to compare (accuracy, latency, etc.)

        Returns:
            test_id: Unique test identifier
        """
        test_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        test_dir = self.output_dir / test_id
        test_dir.mkdir(parents=True, exist_ok=True)

        test_config = {
            'test_id': test_id,
            'test_name': test_name,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'metric': metric,
            'created_at': datetime.now().isoformat()
        }

        with open(test_dir / "config.json", 'w') as f:
            json.dump(test_config, f, indent=2)

        logger.info(f"Created A/B test: {test_id}")
        logger.info(f"  Variant A: {variant_a['name']}")
        logger.info(f"  Variant B: {variant_b['name']}")
        logger.info(f"  Metric: {metric}")

        return test_id

    def run_test(
        self,
        test_id: str,
        prediction_function: Callable,
        test_data: List[Dict],
        min_sample_size: int = 30
    ) -> ABTestResult:
        """
        Run A/B test

        Args:
            test_id: Test identifier
            prediction_function: Function that runs predictions
            test_data: Test dataset
            min_sample_size: Minimum sample size per variant

        Returns:
            ABTestResult
        """
        test_dir = self.output_dir / test_id

        # Load config
        with open(test_dir / "config.json", 'r') as f:
            config = json.load(f)

        variant_a_config = config['variant_a']
        variant_b_config = config['variant_b']

        logger.info(f"Running A/B test: {test_id}")
        logger.info(f"  Test data size: {len(test_data)}")

        # Split data randomly
        np.random.shuffle(test_data)
        split_point = len(test_data) // 2

        variant_a_data = test_data[:split_point]
        variant_b_data = test_data[split_point:]

        # Run variant A
        logger.info(f"Running Variant A: {variant_a_config['name']}...")
        variant_a_results = []
        for data_point in variant_a_data:
            result = prediction_function(data_point, variant_a_config)
            variant_a_results.append(result[config['metric']])

        # Run variant B
        logger.info(f"Running Variant B: {variant_b_config['name']}...")
        variant_b_results = []
        for data_point in variant_b_data:
            result = prediction_function(data_point, variant_b_config)
            variant_b_results.append(result[config['metric']])

        # Statistical analysis
        analysis = self._analyze_results(
            variant_a_results,
            variant_b_results,
            variant_a_config['name'],
            variant_b_config['name']
        )

        # Create result object
        result = ABTestResult(
            test_name=config['test_name'],
            variant_a=Variant(
                name=variant_a_config['name'],
                config=variant_a_config,
                results=variant_a_results,
                sample_size=len(variant_a_results)
            ),
            variant_b=Variant(
                name=variant_b_config['name'],
                config=variant_b_config,
                results=variant_b_results,
                sample_size=len(variant_b_results)
            ),
            winner=analysis['winner'],
            p_value=analysis['p_value'],
            confidence_level=analysis['confidence_level'],
            effect_size=analysis['effect_size'],
            is_significant=analysis['is_significant'],
            timestamp=datetime.now()
        )

        # Save results
        self._save_results(test_id, result)

        return result

    def _analyze_results(
        self,
        results_a: List[float],
        results_b: List[float],
        name_a: str,
        name_b: str
    ) -> Dict:
        """
        Statistical analysis of A/B test results

        Uses t-test for significance testing
        """
        # Calculate means
        mean_a = np.mean(results_a)
        mean_b = np.mean(results_b)

        # T-test
        t_stat, p_value = stats.ttest_ind(results_a, results_b)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(results_a, ddof=1)**2 + np.std(results_b, ddof=1)**2) / 2
        )
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

        # Determine winner
        is_significant = p_value < 0.05
        confidence_level = 1 - p_value

        if not is_significant:
            winner = None
        elif mean_a > mean_b:
            winner = name_a
        else:
            winner = name_b

        logger.info(f"Analysis complete:")
        logger.info(f"  {name_a}: {mean_a:.4f}")
        logger.info(f"  {name_b}: {mean_b:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Significant: {is_significant}")
        logger.info(f"  Winner: {winner or 'None (no significant difference)'}")

        return {
            'mean_a': mean_a,
            'mean_b': mean_b,
            'p_value': p_value,
            'confidence_level': confidence_level,
            'effect_size': effect_size,
            'is_significant': is_significant,
            'winner': winner
        }

    def _save_results(self, test_id: str, result: ABTestResult):
        """Save test results"""
        test_dir = self.output_dir / test_id

        result_dict = {
            'test_name': result.test_name,
            'variant_a': {
                'name': result.variant_a.name,
                'config': result.variant_a.config,
                'mean': float(np.mean(result.variant_a.results)),
                'std': float(np.std(result.variant_a.results)),
                'sample_size': result.variant_a.sample_size
            },
            'variant_b': {
                'name': result.variant_b.name,
                'config': result.variant_b.config,
                'mean': float(np.mean(result.variant_b.results)),
                'std': float(np.std(result.variant_b.results)),
                'sample_size': result.variant_b.sample_size
            },
            'winner': result.winner,
            'p_value': float(result.p_value),
            'confidence_level': float(result.confidence_level),
            'effect_size': float(result.effect_size),
            'is_significant': result.is_significant,
            'timestamp': result.timestamp.isoformat()
        }

        with open(test_dir / "results.json", 'w') as f:
            json.dump(result_dict, f, indent=2)

        logger.info(f"Saved results to: {test_dir / 'results.json'}")

    def generate_report(self, test_id: str) -> str:
        """
        Generate A/B test report

        Args:
            test_id: Test identifier

        Returns:
            Report markdown string
        """
        test_dir = self.output_dir / test_id

        with open(test_dir / "results.json", 'r') as f:
            results = json.load(f)

        report = f"""# A/B Test Report

**Test**: {results['test_name']}
**Date**: {results['timestamp']}

## Results

| Variant | Mean | Std Dev | Sample Size |
|---------|------|---------|-------------|
| **{results['variant_a']['name']}** | {results['variant_a']['mean']:.4f} | {results['variant_a']['std']:.4f} | {results['variant_a']['sample_size']} |
| **{results['variant_b']['name']}** | {results['variant_b']['mean']:.4f} | {results['variant_b']['std']:.4f} | {results['variant_b']['sample_size']} |

## Statistical Analysis

- **P-value**: {results['p_value']:.4f}
- **Confidence Level**: {results['confidence_level']:.1%}
- **Effect Size (Cohen's d)**: {results['effect_size']:.4f}
- **Statistically Significant**: {'✅ YES' if results['is_significant'] else '❌ NO'}

## Winner

"""

        if results['winner']:
            improvement = abs(results['variant_a']['mean'] - results['variant_b']['mean'])
            improvement_pct = (improvement / min(results['variant_a']['mean'], results['variant_b']['mean'])) * 100

            report += f"""
**🏆 {results['winner']}**

- Improvement: {improvement:.4f} ({improvement_pct:.1f}%)
- Significant at p < 0.05 level
"""
        else:
            report += """
**No Clear Winner**

- No statistically significant difference detected
- Both variants perform similarly
- Consider running more samples or testing different configurations
"""

        report += f"""

## Recommendation

"""

        if results['is_significant'] and results['winner']:
            report += f"""
✅ **Deploy {results['winner']}**

The {results['winner']} variant shows statistically significant improvement.
Recommended for production deployment.
"""
        else:
            report += """
⏸️ **Continue Testing**

No significant difference detected. Consider:
1. Increasing sample size
2. Testing different metrics
3. Trying alternative configurations
"""

        return report


# Common A/B tests
def test_cot_strategies():
    """Test CoT strategy comparison"""
    framework = ABTestingFramework()

    test_id = framework.create_test(
        test_name="CoT Strategy Comparison",
        variant_a={'name': 'Sim-CoT', 'cot_strategy': 'sim_cot'},
        variant_b={'name': 'Path-CoT', 'cot_strategy': 'path_cot'},
        metric="accuracy"
    )

    return test_id


def test_retrieval_methods():
    """Test retrieval method comparison"""
    framework = ABTestingFramework()

    test_id = framework.create_test(
        test_name="Retrieval Method Comparison",
        variant_a={'name': 'Hybrid', 'method': 'hybrid', 'weights': [0.4, 0.3, 0.3]},
        variant_b={'name': 'Vector Only', 'method': 'vector', 'weights': [1.0, 0.0, 0.0]},
        metric="recall_at_10"
    )

    return test_id
