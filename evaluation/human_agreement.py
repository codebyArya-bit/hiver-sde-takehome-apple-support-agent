"""
Human-Judge Agreement Analysis.
Measures inter-rater agreement between the automated LLM Judge and Human Annotator scores
on the Golden Evaluation Set using:
- Cohen's Kappa (discrete classification agreement)
- Pearson Correlation Coefficient (linear alignment)
- Spearman Rank Correlation (monotonic ordering alignment)
- Mean Absolute Error (MAE)
- Percentage Agreement (exact and within-0.5 / within-1.0 point tolerance)
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Any
from sklearn.metrics import cohen_kappa_score

def calculate_human_judge_agreement(
    human_scores: List[float],
    judge_scores: List[float],
    metric_name: str = "Overall Score"
) -> Dict[str, Any]:
    """
    Computes statistical agreement metrics between human ground truth and judge evaluations.
    """
    h_arr = np.array(human_scores)
    j_arr = np.array(judge_scores)

    # 1. Mean Absolute Error & Root Mean Squared Error
    mae = float(np.mean(np.abs(h_arr - j_arr)))
    rmse = float(np.sqrt(np.mean((h_arr - j_arr) ** 2)))

    # 2. Pearson Correlation
    pearson_r, p_val = stats.pearsonr(h_arr, j_arr)

    # 3. Spearman Rank Correlation
    spearman_rho, sp_val = stats.spearmanr(h_arr, j_arr)

    # 4. Tolerance Agreement
    diffs = np.abs(h_arr - j_arr)
    within_half_point = float(np.mean(diffs <= 0.5) * 100)
    within_one_point = float(np.mean(diffs <= 1.0) * 100)

    # 5. Cohen's Kappa on Binned Tiers:
    # Low (< 3.0), Moderate (3.0 - 4.2), High (> 4.2)
    def bin_score(s):
        if s < 3.0:
            return "LOW"
        elif s <= 4.2:
            return "MODERATE"
        return "HIGH"

    h_bins = [bin_score(s) for s in human_scores]
    j_bins = [bin_score(s) for s in judge_scores]
    kappa = cohen_kappa_score(h_bins, j_bins)

    return {
        "metric_name": metric_name,
        "sample_size": len(human_scores),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p_value": float(p_val),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p_value": float(sp_val),
        "within_0.5_points_pct": round(within_half_point, 2),
        "within_1.0_points_pct": round(within_one_point, 2),
        "cohens_kappa": round(float(kappa), 4)
    }

def run_comprehensive_agreement_study(
    golden_eval_set: List[Dict[str, Any]],
    judge_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs multi-criteria human-judge agreement analysis across all 4 rubric dimensions.
    """
    dimensions = [
        ("groundedness", "Groundedness & Correctness"),
        ("brand_voice", "Brand Voice & Empathy"),
        ("actionability", "Actionability & Quality"),
        ("escalation_appropriateness", "Escalation Appropriateness"),
        ("overall_score", "Overall Rubric Score")
    ]

    agreement_summary = {}

    for dim_key, dim_title in dimensions:
        h_scores = []
        j_scores = []

        for i, gold in enumerate(golden_eval_set):
            # Human gold rubric
            if dim_key == "overall_score":
                h_val = np.mean([
                    gold["human_rubric"]["groundedness"],
                    gold["human_rubric"]["brand_voice"],
                    gold["human_rubric"]["actionability"],
                    gold["human_rubric"]["escalation_appropriateness"]
                ])
            else:
                h_val = gold["human_rubric"][dim_key]

            # Judge score
            j_val = judge_results["individual_scores"][i][dim_key]

            h_scores.append(float(h_val))
            j_scores.append(float(j_val))

        stats_res = calculate_human_judge_agreement(h_scores, j_scores, metric_name=dim_title)
        agreement_summary[dim_key] = stats_res

    return agreement_summary
