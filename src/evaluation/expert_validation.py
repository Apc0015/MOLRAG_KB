"""
Phase 6: Expert Validation Pipeline
Automated expert review collection and analysis
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExpertReview:
    """Expert review of a prediction"""
    prediction_id: str
    smiles: str
    query: str
    prediction: str
    reasoning: str
    pathways: List[str]
    citations: List[str]

    # Expert ratings (1-5 scale)
    accuracy_rating: int
    reasoning_quality: int
    pathway_relevance: int
    citation_accuracy: int

    # Expert feedback
    feedback: str
    approved: bool
    expert_id: str
    timestamp: datetime


class ExpertValidationPipeline:
    """
    Expert Validation Pipeline

    Collects and analyzes expert reviews of MolRAG predictions.
    Target: 75%+ expert approval rate
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize validation pipeline

        Args:
            output_dir: Directory to store reviews
        """
        self.output_dir = output_dir or Path("expert_reviews")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_review_batch(
        self,
        predictions: List[Dict],
        batch_name: str
    ) -> str:
        """
        Create a batch of predictions for expert review

        Args:
            predictions: List of prediction dicts
            batch_name: Name for this batch

        Returns:
            batch_id: Unique batch identifier
        """
        batch_id = f"{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_dir = self.output_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        batch_file = batch_dir / "predictions.json"
        with open(batch_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Created review batch: {batch_id} with {len(predictions)} predictions")
        return batch_id

    def submit_review(
        self,
        review: ExpertReview,
        batch_id: str
    ):
        """
        Submit expert review

        Args:
            review: ExpertReview object
            batch_id: Batch identifier
        """
        review_file = self.output_dir / batch_id / f"review_{review.prediction_id}.json"

        review_dict = {
            'prediction_id': review.prediction_id,
            'smiles': review.smiles,
            'query': review.query,
            'prediction': review.prediction,
            'reasoning': review.reasoning,
            'pathways': review.pathways,
            'citations': review.citations,
            'accuracy_rating': review.accuracy_rating,
            'reasoning_quality': review.reasoning_quality,
            'pathway_relevance': review.pathway_relevance,
            'citation_accuracy': review.citation_accuracy,
            'feedback': review.feedback,
            'approved': review.approved,
            'expert_id': review.expert_id,
            'timestamp': review.timestamp.isoformat()
        }

        with open(review_file, 'w') as f:
            json.dump(review_dict, f, indent=2)

        logger.info(f"Submitted review for prediction {review.prediction_id}")

    def analyze_reviews(self, batch_id: str) -> Dict:
        """
        Analyze expert reviews for a batch

        Args:
            batch_id: Batch identifier

        Returns:
            Analysis results
        """
        batch_dir = self.output_dir / batch_id
        review_files = list(batch_dir.glob("review_*.json"))

        if not review_files:
            logger.warning(f"No reviews found for batch {batch_id}")
            return {}

        reviews = []
        for review_file in review_files:
            with open(review_file, 'r') as f:
                reviews.append(json.load(f))

        # Calculate metrics
        total_reviews = len(reviews)
        approved = sum(1 for r in reviews if r['approved'])
        approval_rate = approved / total_reviews if total_reviews > 0 else 0

        avg_accuracy = sum(r['accuracy_rating'] for r in reviews) / total_reviews
        avg_reasoning = sum(r['reasoning_quality'] for r in reviews) / total_reviews
        avg_pathway = sum(r['pathway_relevance'] for r in reviews) / total_reviews
        avg_citation = sum(r['citation_accuracy'] for r in reviews) / total_reviews

        analysis = {
            'batch_id': batch_id,
            'total_reviews': total_reviews,
            'approved_count': approved,
            'approval_rate': approval_rate,
            'avg_accuracy_rating': avg_accuracy,
            'avg_reasoning_quality': avg_reasoning,
            'avg_pathway_relevance': avg_pathway,
            'avg_citation_accuracy': avg_citation,
            'target_approval_rate': 0.75,
            'meets_target': approval_rate >= 0.75
        }

        logger.info(f"Batch {batch_id} analysis:")
        logger.info(f"  Approval rate: {approval_rate:.1%} (target: 75%)")
        logger.info(f"  Avg accuracy: {avg_accuracy:.2f}/5")
        logger.info(f"  Avg reasoning: {avg_reasoning:.2f}/5")

        # Save analysis
        analysis_file = batch_dir / "analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        return analysis

    def get_low_rated_predictions(self, batch_id: str, threshold: int = 3) -> List[Dict]:
        """
        Get predictions with low expert ratings

        Args:
            batch_id: Batch identifier
            threshold: Rating threshold (1-5)

        Returns:
            List of low-rated predictions
        """
        batch_dir = self.output_dir / batch_id
        review_files = list(batch_dir.glob("review_*.json"))

        low_rated = []
        for review_file in review_files:
            with open(review_file, 'r') as f:
                review = json.load(f)

            avg_rating = (
                review['accuracy_rating'] +
                review['reasoning_quality'] +
                review['pathway_relevance'] +
                review['citation_accuracy']
            ) / 4

            if avg_rating < threshold:
                low_rated.append(review)

        return low_rated

    def generate_report(self, batch_id: str) -> str:
        """
        Generate expert validation report

        Args:
            batch_id: Batch identifier

        Returns:
            Report markdown string
        """
        analysis = self.analyze_reviews(batch_id)

        if not analysis:
            return "No reviews found"

        report = f"""# Expert Validation Report

**Batch**: {batch_id}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Reviews**: {analysis['total_reviews']}
- **Approved**: {analysis['approved_count']} ({analysis['approval_rate']:.1%})
- **Target Approval Rate**: 75%
- **Meets Target**: {'✅ YES' if analysis['meets_target'] else '❌ NO'}

## Quality Metrics (1-5 scale)

| Metric | Average |
|--------|---------|
| Prediction Accuracy | {analysis['avg_accuracy_rating']:.2f} |
| Reasoning Quality | {analysis['avg_reasoning_quality']:.2f} |
| Pathway Relevance | {analysis['avg_pathway_relevance']:.2f} |
| Citation Accuracy | {analysis['avg_citation_accuracy']:.2f} |

## Recommendations

"""

        if analysis['approval_rate'] < 0.75:
            report += """
### Areas for Improvement

1. **Approval Rate Below Target**
   - Current: {:.1%}
   - Target: 75%
   - Action: Review low-rated predictions and improve model

2. **Focus Areas**
""".format(analysis['approval_rate'])

            if analysis['avg_accuracy_rating'] < 3.5:
                report += "   - Prediction accuracy needs improvement\n"
            if analysis['avg_reasoning_quality'] < 3.5:
                report += "   - Reasoning quality needs improvement\n"
            if analysis['avg_pathway_relevance'] < 3.5:
                report += "   - Pathway relevance needs improvement\n"
            if analysis['avg_citation_accuracy'] < 3.5:
                report += "   - Citation accuracy needs improvement\n"
        else:
            report += """
### Status: ✅ Approved

System meets expert validation criteria with {:.1%} approval rate.
Continue monitoring and maintaining quality standards.
""".format(analysis['approval_rate'])

        return report
