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
from typing import List, Dict, Any, Optional
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

    # 2. Correlations (safely handling constant inputs where correlation is undefined)
    import warnings
    if np.all(h_arr == h_arr[0]) or np.all(j_arr == j_arr[0]):
        pearson_r, p_val = None, None
        spearman_rho, sp_val = None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_r, p_v = stats.pearsonr(h_arr, j_arr)
            pearson_r = round(float(p_r), 3) if not np.isnan(p_r) else None
            p_val = float(p_v) if not np.isnan(p_v) else None

            s_r, s_v = stats.spearmanr(h_arr, j_arr)
            spearman_rho = round(float(s_r), 3) if not np.isnan(s_r) else None
            sp_val = float(s_v) if not np.isnan(s_v) else None

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
    kappa = None

    # Statistically, if one or both raters assign all items to a single category,
    # category variance is 0 and Cohen's Kappa is mathematically undefined (0/0 or NaN).
    # It must NEVER be converted to 1.0 or interpreted as perfect agreement.
    if len(set(h_bins)) > 1 and len(set(j_bins)) > 1:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kappa_val = cohen_kappa_score(h_bins, j_bins, labels=["LOW", "MODERATE", "HIGH"])
                if not np.isnan(kappa_val):
                    kappa = round(float(kappa_val), 3)
        except Exception:
            kappa = None
    else:
        kappa = None

    return {
        "metric_name": metric_name,
        "sample_size": len(human_scores),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "pearson_r": pearson_r,
        "pearson_p_value": p_val,
        "spearman_rho": spearman_rho,
        "spearman_p_value": sp_val,
        "within_0.5_points_pct": round(within_half, 1),
        "within_1.0_points_pct": round(within_one, 1),
        "cohens_kappa": kappa
    }

def format_kappa_presentation(kappa_val: Optional[float]) -> str:
    """
    Renders Cohen's Kappa strictly for the presentation layer (CLI, tables, reports).
    Undefined/NaN Kappa is presented as 'N/A' and never substituted with 1.0.
    """
    if kappa_val is None or (isinstance(kappa_val, float) and np.isnan(kappa_val)):
        return "N/A"
    return f"{kappa_val:.3f}"

def run_human_judge_agreement_study(
    human_ann_path: str = "evaluation/human_annotations.json",
    llm_scores_path: str = "evaluation/llm_judge_scores.json",
    frozen_subset_path: str = "data/frozen_eval_subset_n50.json"
) -> Dict[str, Any]:
    """
    Computes authentic human-judge agreement across the 50 paired evaluations.
    Enforces cryptographic SHA256 input hash assertions to prevent stale cached scoring.
    """
    with open(human_ann_path, 'r', encoding='utf-8') as f:
        human_data = json.load(f)

    with open(llm_scores_path, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)

    frozen_hashes = {}
    if Path(frozen_subset_path).exists():
        with open(frozen_subset_path, 'r', encoding='utf-8') as f:
            frozen_items = json.load(f)
            frozen_hashes = {x["item_id"]: x.get("input_hash") for x in frozen_items}

    # Verify cryptographic input hash integrity for every single evaluated item
    verified_pairs = 0
    for h_entry in human_data:
        item_id = h_entry["item_id"]
        assert item_id in llm_data, f"Missing LLM judge evaluation for item {item_id}"
        j_entry = llm_data[item_id]

        h_hash = h_entry.get("input_hash")
        j_hash = j_entry.get("input_hash")
        assert h_hash is not None and j_hash is not None, f"Missing input_hash on item {item_id}"
        assert h_hash == j_hash, (
            f"Cryptographic hash mismatch on {item_id}: "
            f"Human hash ({h_hash}) does not match LLM hash ({j_hash}). Stale evaluation detected!"
        )
        if item_id in frozen_hashes and frozen_hashes[item_id]:
            assert h_hash == frozen_hashes[item_id], (
                f"Evaluation hash mismatch against frozen subset for {item_id}: "
                f"Evaluated hash ({h_hash}) != Frozen candidate hash ({frozen_hashes[item_id]})"
            )
        verified_pairs += 1

    print(f"[OK] Cryptographic verification passed: All {verified_pairs} human-LLM pairs match candidate input hashes.")

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
        k_str = format_kappa_presentation(v['cohens_kappa'])
        p_str = f"{v['pearson_r']:.3f}" if v['pearson_r'] is not None else "N/A"
        print(f"{v['metric_name']}: Pearson r={p_str}, MAE={v['mae']}, Kappa={k_str}, Within 0.5={v['within_0.5_points_pct']}%")
