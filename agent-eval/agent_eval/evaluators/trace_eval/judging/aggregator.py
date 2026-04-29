# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Aggregation Logic for Judge Results

This module implements within-judge and cross-judge aggregation for evaluation results.
Supports both categorical (majority vote) and numeric (median, mean, variance) scoring.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import statistics
from collections import Counter

from .models import JobResult


# Canonical status definitions - must match runner.py
CANONICAL_SUCCESS_STATUSES = {"success"}
CANONICAL_FAILURE_STATUSES = {"failure", "timeout", "invalid_response", "cancelled", "error", "failed"}


@dataclass
class ScoringScale:
    """
    Represents a rubric's scoring scale for scale-aware aggregation.
    
    For numeric scales: type="numeric", min and max define the range
    For categorical scales: type="categorical", values list the categories
    """
    type: str  # "numeric" or "categorical"
    min: Optional[float] = None  # For numeric scales
    max: Optional[float] = None  # For numeric scales
    values: Optional[List[str]] = None  # For categorical scales
    
    def __post_init__(self):
        """Validate scoring scale configuration."""
        if self.type == "numeric":
            if self.min is None or self.max is None:
                raise ValueError("Numeric scoring scale requires min and max")
            if self.min >= self.max:
                raise ValueError(f"Invalid numeric scale: min ({self.min}) must be < max ({self.max})")
        elif self.type == "categorical":
            if not self.values or len(self.values) == 0:
                raise ValueError("Categorical scoring scale requires non-empty values list")
        else:
            raise ValueError(f"Invalid scoring scale type: {self.type}. Must be 'numeric' or 'categorical'")
    
    def get_range(self) -> float:
        """Get the range for numeric scales."""
        if self.type != "numeric":
            raise ValueError("Range only applicable to numeric scales")
        return self.max - self.min


@dataclass
class WithinJudgeResult:
    """
    Aggregated results from repeated runs of the same judge.
    
    Represents consensus and variance within a single judge's evaluations.
    """
    judge_id: str
    rubric_id: str
    turn_id: Optional[str]  # None for run-level rubrics
    majority_vote: Optional[str] = None  # For categorical scores
    median: Optional[float] = None  # For numeric scores
    mean: Optional[float] = None  # For numeric scores
    variance: float = 0.0  # Consistency signal
    sample_size: int = 0  # Number of successful repeats
    scoring_type: str = "numeric"  # "numeric" or "categorical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for trace_eval.json."""
        return {
            "judge_id": self.judge_id,
            "rubric_id": self.rubric_id,
            "turn_id": self.turn_id,
            "majority_vote": self.majority_vote,
            "median": self.median,
            "mean": self.mean,
            "variance": self.variance,
            "sample_size": self.sample_size,
            "scoring_type": self.scoring_type
        }


@dataclass
class CrossJudgeResult:
    """
    Aggregated results across different judges.
    
    Represents consensus and disagreement across multiple judges.
    """
    rubric_id: str
    turn_id: Optional[str]  # None for run-level rubrics
    weighted_vote: Optional[str] = None  # For categorical
    weighted_average: Optional[float] = None  # For numeric
    disagreement_signal: float = 0.0  # 0.0 to 1.0
    high_risk_flag: bool = False  # True if disagreement > threshold
    judge_count: int = 0  # Total judges with usable results
    scoring_type: str = "numeric"  # "numeric" or "categorical"
    individual_judge_results: List[WithinJudgeResult] = field(default_factory=list)
    mixed_types_error: Optional[str] = None  # Set if scoring types are inconsistent
    scale_warning: Optional[str] = None  # Set if scale-related issues detected
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for trace_eval.json."""
        return {
            "rubric_id": self.rubric_id,
            "turn_id": self.turn_id,
            "weighted_vote": self.weighted_vote,
            "weighted_average": self.weighted_average,
            "disagreement_signal": self.disagreement_signal,
            "high_risk_flag": self.high_risk_flag,
            "judge_count": self.judge_count,
            "scoring_type": self.scoring_type,
            "mixed_types_error": self.mixed_types_error,
            "scale_warning": self.scale_warning,
            "individual_judge_results": [
                result.to_dict() for result in self.individual_judge_results
            ]
        }


class Aggregator:
    """
    Aggregates judge results within-judge and cross-judge.
    
    Supports both categorical (majority vote) and numeric (median, mean, variance) scoring.
    Uses scale-aware disagreement computation for numeric rubrics.
    """
    
    def __init__(self, disagreement_threshold: float = 0.3):
        """
        Initialize aggregator.
        
        Args:
            disagreement_threshold: Threshold for high-risk disagreement flag (default 0.3)
        """
        self.disagreement_threshold = disagreement_threshold
    
    def aggregate_within_judge(
        self,
        results: List[JobResult],
        judge_id: str,
        rubric_id: str,
        turn_id: Optional[str] = None,
        scoring_scale: Optional[ScoringScale] = None
    ) -> WithinJudgeResult:
        """
        Aggregate repeated runs from the same judge.
        
        Computes:
        - Majority vote for categorical scores
        - Median, mean, variance for numeric scores
        
        Args:
            results: List of JobResult from the same judge
            judge_id: Judge identifier
            rubric_id: Rubric identifier
            turn_id: Turn identifier (None for run-level)
            scoring_scale: Optional ScoringScale for scale-normalized variance
            
        Returns:
            WithinJudgeResult with aggregated statistics
        """
        # Filter successful results only - use canonical status definitions
        successful_results = [r for r in results if r.status in CANONICAL_SUCCESS_STATUSES]
        
        if not successful_results:
            return WithinJudgeResult(
                judge_id=judge_id,
                rubric_id=rubric_id,
                turn_id=turn_id,
                sample_size=0
            )
        
        # Extract scores
        scores = [r.score for r in successful_results if r.score is not None]
        
        if not scores:
            # Successful results but no scores - still report sample_size for transparency
            return WithinJudgeResult(
                judge_id=judge_id,
                rubric_id=rubric_id,
                turn_id=turn_id,
                sample_size=len(successful_results)  # Successful repeats, but no scores
            )
        
        # Determine scoring type
        scoring_type = self._determine_scoring_type(scores)
        
        if scoring_type == "categorical":
            return self._aggregate_categorical(
                scores, judge_id, rubric_id, turn_id, len(successful_results)
            )
        else:
            return self._aggregate_numeric(
                scores, judge_id, rubric_id, turn_id, len(successful_results), scoring_scale
            )
    
    def _determine_scoring_type(self, scores: List[Union[float, str]]) -> str:
        """
        Determine if scores are numeric or categorical.
        
        Attempts to coerce strings to floats to handle numeric strings.
        
        Args:
            scores: List of scores
            
        Returns:
            "numeric" or "categorical"
        """
        # Try to coerce all scores to float
        numeric_count = 0
        for s in scores:
            try:
                float(s)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        # If all scores can be converted to float, treat as numeric
        if numeric_count == len(scores):
            return "numeric"
        return "categorical"
    
    def _aggregate_categorical(
        self,
        scores: List[str],
        judge_id: str,
        rubric_id: str,
        turn_id: Optional[str],
        sample_size: int
    ) -> WithinJudgeResult:
        """
        Aggregate categorical scores using majority vote.
        
        Args:
            scores: List of categorical scores
            judge_id: Judge identifier
            rubric_id: Rubric identifier
            turn_id: Turn identifier
            sample_size: Number of successful results
            
        Returns:
            WithinJudgeResult with majority_vote
        """
        # Convert to strings for consistency
        str_scores = [str(s) for s in scores]
        
        # Compute majority vote
        vote_counts = Counter(str_scores)
        majority_vote = vote_counts.most_common(1)[0][0]
        
        # Compute disagreement ratio (stored as variance for backward compatibility)
        # disagreement_ratio = 1 - (max_count / total_count)
        max_count = vote_counts.most_common(1)[0][1]
        disagreement_ratio = 1.0 - (max_count / len(str_scores)) if str_scores else 0.0
        
        return WithinJudgeResult(
            judge_id=judge_id,
            rubric_id=rubric_id,
            turn_id=turn_id,
            majority_vote=majority_vote,
            variance=disagreement_ratio,  # Field name kept for backward compatibility
            sample_size=sample_size,
            scoring_type="categorical"
        )
    
    def _aggregate_numeric(
        self,
        scores: List[float],
        judge_id: str,
        rubric_id: str,
        turn_id: Optional[str],
        sample_size: int,
        scoring_scale: Optional[ScoringScale] = None
    ) -> WithinJudgeResult:
        """
        Aggregate numeric scores using median, mean, variance.
        
        If scoring_scale provided, normalizes variance by scale range for
        cross-rubric comparability.
        
        Args:
            scores: List of numeric scores
            judge_id: Judge identifier
            rubric_id: Rubric identifier
            turn_id: Turn identifier
            sample_size: Number of successful results
            scoring_scale: Optional ScoringScale for normalized variance
            
        Returns:
            WithinJudgeResult with median, mean, variance
        """
        # Convert to floats for consistency
        float_scores = [float(s) for s in scores]
        
        # Compute statistics
        median_score = statistics.median(float_scores)
        mean_score = statistics.mean(float_scores)
        
        # Compute variance
        if len(float_scores) > 1:
            raw_variance = statistics.pvariance(float_scores)
            
            # Normalize variance by scale range if available
            if scoring_scale and scoring_scale.type == "numeric":
                scale_range = scoring_scale.get_range()
                if scale_range > 0:
                    # Normalized variance = variance / (range^2)
                    variance = raw_variance / (scale_range ** 2)
                else:
                    variance = raw_variance
            else:
                variance = raw_variance
        else:
            variance = 0.0
        
        return WithinJudgeResult(
            judge_id=judge_id,
            rubric_id=rubric_id,
            turn_id=turn_id,
            median=median_score,
            mean=mean_score,
            variance=variance,
            sample_size=sample_size,
            scoring_type="numeric"
        )
    
    def aggregate_cross_judge(
        self,
        within_judge_results: List[WithinJudgeResult],
        rubric_id: str,
        turn_id: Optional[str] = None,
        scoring_scale: Optional[ScoringScale] = None
    ) -> CrossJudgeResult:
        """
        Aggregate results across different judges.
        
        Computes:
        - Weighted vote for categorical scores (weighted by sample_size)
        - Weighted average for numeric scores (weighted by sample_size and inverse variance)
        - Disagreement signal (scale-aware for numeric)
        - High-risk flag if disagreement exceeds threshold
        
        Args:
            within_judge_results: List of WithinJudgeResult from different judges
            rubric_id: Rubric identifier
            turn_id: Turn identifier (None for run-level)
            scoring_scale: Optional ScoringScale for scale-aware disagreement
            
        Returns:
            CrossJudgeResult with aggregated statistics
        """
        if not within_judge_results:
            return CrossJudgeResult(
                rubric_id=rubric_id,
                turn_id=turn_id,
                judge_count=0,
                individual_judge_results=[]
            )
        
        # Filter to judges with usable results (sample_size > 0)
        usable_results = [r for r in within_judge_results if r.sample_size > 0]
        
        if not usable_results:
            return CrossJudgeResult(
                rubric_id=rubric_id,
                turn_id=turn_id,
                judge_count=0,
                individual_judge_results=within_judge_results
            )
        
        # Validate scoring type consistency across all usable judges
        scoring_types = {r.scoring_type for r in usable_results}
        if len(scoring_types) > 1:
            # Mixed types detected - mark as high-risk error
            # Keep scoring_type as the most common type (or first) to maintain schema contract
            type_counts = Counter([r.scoring_type for r in usable_results])
            dominant_type = type_counts.most_common(1)[0][0]
            
            return CrossJudgeResult(
                rubric_id=rubric_id,
                turn_id=turn_id,
                judge_count=len(usable_results),
                scoring_type=dominant_type,  # Use dominant type, not "mixed"
                disagreement_signal=1.0,
                high_risk_flag=True,
                individual_judge_results=within_judge_results,
                mixed_types_error=f"Inconsistent scoring types across judges: {scoring_types}"
            )
        
        # All usable results have consistent scoring type
        scoring_type = usable_results[0].scoring_type
        
        if scoring_type == "categorical":
            return self._aggregate_cross_judge_categorical(
                usable_results, rubric_id, turn_id
            )
        else:
            return self._aggregate_cross_judge_numeric(
                usable_results, rubric_id, turn_id, scoring_scale
            )
    
    def _aggregate_cross_judge_categorical(
        self,
        within_judge_results: List[WithinJudgeResult],
        rubric_id: str,
        turn_id: Optional[str]
    ) -> CrossJudgeResult:
        """
        Aggregate categorical scores across judges using sample-size-weighted vote.
        
        Args:
            within_judge_results: List of WithinJudgeResult (all with sample_size > 0)
            rubric_id: Rubric identifier
            turn_id: Turn identifier
            
        Returns:
            CrossJudgeResult with weighted_vote and disagreement
        """
        # Build weighted vote counts directly (avoid memory explosion)
        vote_counts: Dict[str, int] = {}
        usable_judge_count = 0
        for r in within_judge_results:
            if r.majority_vote is not None:
                vote_counts[r.majority_vote] = vote_counts.get(r.majority_vote, 0) + r.sample_size
                usable_judge_count += 1
        
        if not vote_counts:
            return CrossJudgeResult(
                rubric_id=rubric_id,
                turn_id=turn_id,
                judge_count=0,  # No usable judges
                scoring_type="categorical",
                individual_judge_results=within_judge_results
            )
        
        # Compute weighted vote (category with highest total weight)
        weighted_vote = max(vote_counts.items(), key=lambda x: x[1])[0]
        
        # Compute disagreement signal using unweighted judge votes
        judge_votes = [r.majority_vote for r in within_judge_results if r.majority_vote is not None]
        disagreement, high_risk = self.compute_disagreement(
            judge_votes, threshold=self.disagreement_threshold
        )
        
        return CrossJudgeResult(
            rubric_id=rubric_id,
            turn_id=turn_id,
            weighted_vote=weighted_vote,
            disagreement_signal=disagreement,
            high_risk_flag=high_risk,
            judge_count=usable_judge_count,  # Count of judges with usable votes
            scoring_type="categorical",
            individual_judge_results=within_judge_results
        )
    
    def _aggregate_cross_judge_numeric(
        self,
        within_judge_results: List[WithinJudgeResult],
        rubric_id: str,
        turn_id: Optional[str],
        scoring_scale: Optional[ScoringScale] = None
    ) -> CrossJudgeResult:
        """
        Aggregate numeric scores across judges using variance-weighted average.
        
        Weights each judge by sample_size and inverse variance (more consistent judges
        get higher weight). Filters out judges with non-finite or negative variance.
        
        Args:
            within_judge_results: List of WithinJudgeResult (all with sample_size > 0)
            rubric_id: Rubric identifier
            turn_id: Turn identifier
            scoring_scale: Optional ScoringScale for scale-aware disagreement
            
        Returns:
            CrossJudgeResult with weighted_average and disagreement
        """
        # Extract medians with weights, filtering non-finite variance
        import math
        judge_data = []
        for r in within_judge_results:
            if r.median is not None:
                # Clamp variance for weighting (must be non-negative and finite)
                variance_for_weight = r.variance
                if not math.isfinite(variance_for_weight) or variance_for_weight < 0:
                    variance_for_weight = 0.0
                
                # Weight = sample_size / (1 + variance)
                # Higher sample_size and lower variance = higher weight
                weight = r.sample_size / (1.0 + variance_for_weight)
                
                # Only include if weight is finite
                if math.isfinite(weight):
                    judge_data.append({
                        'median': r.median,
                        'weight': weight,
                        'variance': r.variance
                    })
        
        if not judge_data:
            return CrossJudgeResult(
                rubric_id=rubric_id,
                turn_id=turn_id,
                judge_count=0,  # No usable judges with valid medians
                scoring_type="numeric",
                individual_judge_results=within_judge_results
            )
        
        # Compute weighted average
        total_weight = sum(d['weight'] for d in judge_data)
        
        # Guard against zero/invalid total weight
        if not math.isfinite(total_weight) or total_weight <= 0:
            # Fallback to simple mean of medians
            medians = [d['median'] for d in judge_data]
            weighted_average = statistics.mean(medians)
            scale_warning = f"Invalid total weight ({total_weight}), using simple mean fallback"
        else:
            weighted_average = sum(d['median'] * d['weight'] for d in judge_data) / total_weight
            scale_warning = None
        
        # Compute disagreement signal (scale-aware if scoring_scale provided)
        medians = [d['median'] for d in judge_data]
        disagreement, high_risk = self.compute_disagreement(
            medians, 
            threshold=self.disagreement_threshold,
            scoring_scale=scoring_scale
        )
        
        return CrossJudgeResult(
            rubric_id=rubric_id,
            turn_id=turn_id,
            weighted_average=weighted_average,
            disagreement_signal=disagreement,
            high_risk_flag=high_risk,
            judge_count=len(judge_data),  # Count of judges with usable medians and finite weights
            scoring_type="numeric",
            individual_judge_results=within_judge_results,
            scale_warning=scale_warning
        )
    
    def compute_disagreement(
        self,
        scores: List[Union[float, str]],
        threshold: float = 0.3,
        scoring_scale: Optional[ScoringScale] = None
    ) -> tuple[float, bool]:
        """
        Compute disagreement signal and high-risk flag.
        
        For numeric scores: Uses scale-aware normalized standard deviation
        For categorical scores: Uses entropy-based disagreement
        
        Args:
            scores: List of scores (numeric or categorical)
            threshold: Threshold for high-risk flag (default 0.3)
            scoring_scale: Optional ScoringScale for scale-aware numeric disagreement
            
        Returns:
            Tuple of (disagreement_score, high_risk_flag)
            - disagreement_score: 0.0 to 1.0
            - high_risk_flag: True if disagreement > threshold
        """
        if not scores or len(scores) < 2:
            return 0.0, False
        
        # Determine if numeric or categorical (consistent with _determine_scoring_type)
        numeric_count = 0
        for s in scores:
            try:
                float(s)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        # If all scores can be converted to float, treat as numeric
        if numeric_count == len(scores):
            disagreement = self._compute_numeric_disagreement(scores, scoring_scale)
        else:
            disagreement = self._compute_categorical_disagreement(scores)
        
        high_risk = disagreement > threshold
        
        return disagreement, high_risk
    
    def _compute_numeric_disagreement(
        self, 
        scores: List[float],
        scoring_scale: Optional[ScoringScale] = None
    ) -> float:
        """
        Compute disagreement for numeric scores using scale-aware normalized std deviation.
        
        If scoring_scale is provided, normalizes by the scale range (max - min).
        Otherwise, uses actual score range or coefficient of variation as fallback.
        
        Args:
            scores: List of numeric scores
            scoring_scale: Optional ScoringScale for scale-aware normalization
            
        Returns:
            Disagreement score (0.0 to 1.0)
        """
        float_scores = [float(s) for s in scores]
        
        if len(float_scores) < 2:
            return 0.0
        
        std_dev = statistics.stdev(float_scores)
        
        # Scale-aware disagreement (preferred)
        if scoring_scale and scoring_scale.type == "numeric":
            scale_range = scoring_scale.get_range()
            if scale_range > 0:
                # Normalize by scale range
                normalized_disagreement = std_dev / scale_range
                return min(normalized_disagreement, 1.0)
        
        # Fallback 1: Use actual score range if available
        score_min = min(float_scores)
        score_max = max(float_scores)
        score_range = score_max - score_min
        
        if score_range > 0:
            # Normalize by actual observed range
            return min(std_dev / score_range, 1.0)
        
        # Fallback 2: Coefficient of variation (only if mean is not near zero)
        mean_score = statistics.mean(float_scores)
        
        if abs(mean_score) >= 0.01:
            # Coefficient of variation: std / |mean|
            cv = std_dev / abs(mean_score)
            return min(cv, 1.0)
        
        # Fallback 3: All scores are identical or near zero
        # Return 0 disagreement (perfect agreement)
        return 0.0
    
    def _get_scale_warning_if_needed(
        self,
        scoring_scale: Optional[ScoringScale],
        fallback_used: str
    ) -> Optional[str]:
        """
        Generate scale warning if scale-aware computation was not possible.
        
        Args:
            scoring_scale: The scoring scale provided (or None)
            fallback_used: Description of which fallback was used
            
        Returns:
            Warning message or None
        """
        if scoring_scale is None:
            return f"No scoring_scale provided, using {fallback_used}"
        elif scoring_scale.type == "numeric" and scoring_scale.get_range() <= 0:
            return f"Invalid scoring_scale range ({scoring_scale.get_range()}), using {fallback_used}"
        return None
    
    def _compute_categorical_disagreement(self, scores: List[str]) -> float:
        """
        Compute disagreement for categorical scores using entropy.
        
        Args:
            scores: List of categorical scores
            
        Returns:
            Disagreement score (0.0 to 1.0)
        """
        str_scores = [str(s) for s in scores]
        
        # Count occurrences
        counts = Counter(str_scores)
        total = len(str_scores)
        
        # Compute normalized entropy
        # entropy = -sum(p * log2(p)) for each category
        # max_entropy = log2(num_categories)
        import math
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize by max possible entropy
        num_categories = len(counts)
        max_entropy = math.log2(num_categories) if num_categories > 1 else 1.0
        
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
