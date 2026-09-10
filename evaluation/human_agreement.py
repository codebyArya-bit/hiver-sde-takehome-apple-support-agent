"""
Human-Judge Agreement Analysis.
Measures inter-rater agreement between the automated LLM Judge and Human Annotator scores
evaluating the EXACT SAME proposed-agent responses across 50 holdout items using:
- Pearson Correlation Coefficient (linear alignment)
- Spearman Rank Correlation (monotonic ordering alignment)
- Mean Absolute Error (MAE on 1.0 - 5.0 raw scale)
- Tolerance Agreement (% within 0.5 points)
- Cohen's Kappa (discrete categorical agreement across quality tiers)
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

def calculate_agreement_metrics(
    human_scores: List[float],
    judge_scores: List[float],
    metric_name: str = "Overall Quality"
) -> Dict[str, Any]:
    h_arr = np.array(human_scores)
    j_arr = np.array(judge_scores)

    # 1. Error metrics
    mae = float(np.mean(np.abs(h_arr - j_arr)))
    rmse = float(np.sqrt(np.mean((h_arr - j_arr) ** 2)))

    # 2. Correlations
    pearson_r, p_val = stats.pearsonr(h_arr, j_arr)
    spearman_rho, sp_val = stats.spearmanr(h_arr, j_arr)

    # 3. Tolerance Agreement
    diffs = np.abs(h_arr - j_arr)
    within_half = float(np.mean(diffs <= 0.5) * 100)
    within_one = float(np.mean(diffs <= 1.0) * 100)

    # 4. Cohen's Kappa on Quality Tiers: Low (<3.5), Moderate (3.5 - 4.5), High (>4.5)
    def bin_score(s):
        if s < 3.5:
            return "LOW"
        elif s <= 4.5:
            return "MODERATE"
        return "HIGH"

    h_bins = [bin_score(s) for s in human_scores]
    j_bins = [bin_score(s) for s in judge_scores]
    kappa = cohen_kappa_score(h_bins, j_bins)

    return {
        "metric_name": metric_name,
        "sample_size": len(human_scores),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "pearson_r": round(float(pearson_r), 3),
        "pearson_p_value": float(p_val),
        "spearman_rho": round(float(spearman_rho), 3),
        "spearman_p_value": float(sp_val),
        "within_0.5_points_pct": round(within_half, 1),
        "within_1.0_points_pct": round(within_one, 1),
        "cohens_kappa": round(float(kappa), 3)
    }

def run_human_judge_agreement_study(
    human_ann_path: str = "evaluation/human_annotations.json",
    llm_scores_path: str = "evaluation/llm_judge_scores.json"
) -> Dict[str, Any]:
    """
    Computes authentic human-judge agreement across the 50 paired evaluations.
    """
    with open(human_ann_path, 'r', encoding='utf-8') as f:
        human_data = json.load(f)

    with open(llm_scores_path, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)

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

        for h_entry in human_data:
            item_id = h_entry["item_id"]
            if item_id in llm_data:
                j_entry = llm_data[item_id]
                if dim_key == "overall_score":
                    h_val = h_entry["overall_score"]
                    j_val = j_entry["overall_score"]
                else:
                    h_val = h_entry["scores"][dim_key]
                    j_val = j_entry["scores"][dim_key]

                h_scores.append(float(h_val))
                j_scores.append(float(j_val))

        agreement_summary[dim_key] = calculate_agreement_metrics(
            h_scores, j_scores, metric_name=dim_title
        )

    return agreement_summary

if __name__ == '__main__':
    res = run_human_judge_agreement_study()
    for k, v in res.items():
        print(f"{v['metric_name']}: Pearson r={v['pearson_r']}, MAE={v['mae']}, Kappa={v['cohens_kappa']}, Within 0.5={v['within_0.5_points_pct']}%")
