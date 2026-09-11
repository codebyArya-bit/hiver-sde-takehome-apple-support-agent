"""
Automated evaluation metrics for Apple Support Agent.
Calculates Intent Classification metrics, Escalation Triage metrics,
and Reply Generation lexical/semantic overlap metrics.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import sacrebleu

# Whitelisted official Apple Support domains
OFFICIAL_APPLE_DOMAINS = [
    "support.apple.com",
    "reportaproblem.apple.com",
    "iforgot.apple.com",
    "apple.com",
    "apple.co"
]

# Mapping from intent to expected authoritative domain anchors
INTENT_EXPECTED_LINK_KEYWORDS = {
    "APP_STORE_AND_BILLING": ["reportaproblem", "billing", "subscriptions"],
    "APPLE_ID_AND_ICLOUD": ["iforgot", "appleid", "icloud"],
    "BATTERY_AND_HARDWARE": ["repair", "battery", "HT208387"],
    "IOS_SOFTWARE_UPDATE": ["HT204204", "update", "ios"],
    "DEVICE_SETUP_AND_USAGE": ["guide", "iphone", "support"],
    "CUSTOMER_FEEDBACK_COMPLAINT": ["feedback", "messages", "dm"],
    "OUT_OF_SCOPE_OTHER": ["support", "apple"]
}

def calculate_intent_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    y_probs: Optional[List[Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Computes classification accuracy, macro/weighted F1, per-class metrics,
    and statistical calibration metrics (Brier Score & Expected Calibration Error).
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    p_per, r_per, f_per, s_per = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f_per[i]), 4),
            "support": int(s_per[i])
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Statistical Calibration Metrics (Brier Score & ECE)
    brier_score = None
    ece = None
    if y_probs and len(y_probs) == len(y_true):
        # 1. Multi-class Brier Score: sum_k (p_ik - y_ik)^2 / N
        brier_sum = 0.0
        confidences = []
        accuracies = []
        for i, (yt, yp_dict) in enumerate(zip(y_true, y_probs)):
            for lbl in labels:
                prob = float(yp_dict.get(lbl, 0.0))
                y_bin = 1.0 if yt == lbl else 0.0
                brier_sum += (prob - y_bin) ** 2
            top_conf = max(yp_dict.values()) if yp_dict else 0.0
            confidences.append(top_conf)
            accuracies.append(1.0 if yt == y_pred[i] else 0.0)

        brier_score = round(brier_sum / len(y_true), 4)

        # 2. Expected Calibration Error (ECE) across 10 equal bins [0, 1]
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0
        n_samples = len(y_true)

        for b in range(n_bins):
            bin_lower = bin_boundaries[b]
            bin_upper = bin_boundaries[b + 1]
            in_bin = [
                i for i, c in enumerate(confidences)
                if (bin_lower <= c < bin_upper) or (b == n_bins - 1 and bin_lower <= c <= bin_upper)
            ]
            if len(in_bin) > 0:
                bin_acc = np.mean([accuracies[i] for i in in_bin])
                bin_conf = np.mean([confidences[i] for i in in_bin])
                ece_val += (len(in_bin) / n_samples) * abs(bin_acc - bin_conf)

        ece = round(float(ece_val), 4)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "weighted_f1": round(float(f1_weight), 4),
        "brier_score": brier_score,
        "expected_calibration_error": ece,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels
    }

def calculate_escalation_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes safety-critical escalation metrics:
    - Escalation Recall (Safety Critical): % of true human escalation cases caught
    - Escalation Precision: % of escalated cases that actually needed human agents
    - Escalation F2-Score: Recall-weighted harmonic mean (weights recall 2x vs precision)
    - False Escalation Rate: % of auto-handle cases wrongly sent to humans (capacity cost)
    - Weighted Risk-Cost Index: Total penalty = 5 * FN (critical missed risk) + 1 * FP (capacity cost)
    """
    acc = accuracy_score(y_true, y_pred)
    
    true_esc = [1 if y == "ESCALATE" else 0 for y in y_true]
    pred_esc = [1 if y == "ESCALATE" else 0 for y in y_pred]

    p, r, f1, _ = precision_recall_fscore_support(
        true_esc, pred_esc, average='binary', zero_division=0
    )

    # F2 Score: (1 + 2^2) * (P * R) / (4 * P + R)
    beta = 2.0
    if (beta**2 * p + r) > 0:
        f2 = ((1 + beta**2) * p * r) / (beta**2 * p + r)
    else:
        f2 = 0.0

    cm = confusion_matrix(true_esc, pred_esc, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    false_esc_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    weighted_risk_cost = (5 * fn) + (1 * fp)

    return {
        "accuracy": round(float(acc), 4),
        "escalation_recall": round(float(r), 4),
        "escalation_precision": round(float(p), 4),
        "escalation_f1": round(float(f1), 4),
        "escalation_f2": round(float(f2), 4),
        "false_escalation_rate": round(float(false_esc_rate), 4),
        "weighted_risk_cost": int(weighted_risk_cost),
        "true_positives_escalate": int(tp),
        "false_negatives_missed_escalate": int(fn),
        "false_positives_unnecessary_escalate": int(fp),
        "true_negatives_correct_autohandle": int(tn)
    }

def calculate_generation_metrics(
    predictions: List[str],
    references: List[str],
    target_intents: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes text generation metrics: BLEU, token overlap, Twitter constraint compliance,
    and verified official Apple domain inclusion rate.
    """
    # 1. Character length compliance (<280 chars)
    length_compliant = sum(1 for p in predictions if len(p) <= 280)
    compliance_rate = length_compliant / len(predictions) if predictions else 0.0

    # 2. Official Apple Domain Inclusion Rate
    valid_links = 0
    relevant_links = 0
    total_with_links = 0

    for i, p in enumerate(predictions):
        p_lower = p.lower()
        has_official_domain = any(domain in p_lower for domain in OFFICIAL_APPLE_DOMAINS)
        
        if has_official_domain:
            valid_links += 1
            total_with_links += 1
            
            # Check relevance to intent if available
            if target_intents and i < len(target_intents):
                intent = target_intents[i]
                expected_kw = INTENT_EXPECTED_LINK_KEYWORDS.get(intent, ["apple"])
                if any(kw in p_lower for kw in expected_kw):
                    relevant_links += 1
                else:
                    relevant_links += 0.5  # general apple link
        elif "http" in p_lower:
            total_with_links += 1

    validity_rate = valid_links / len(predictions) if predictions else 0.0
    relevance_rate = (relevant_links / valid_links) if valid_links > 0 else 0.0

    # 3. SacreBLEU (expects list of reference streams: [references])
    bleu_score = sacrebleu.corpus_bleu(predictions, [references])

    # 4. Token overlap
    def ngrams(text, n):
        words = text.lower().split()
        return [tuple(words[idx:idx+n]) for idx in range(len(words)-n+1)]

    r1_scores, r2_scores = [], []
    for pred, ref in zip(predictions, references):
        p1 = set(ngrams(pred, 1))
        r1 = set(ngrams(ref, 1))
        if r1:
            r1_scores.append(len(p1 & r1) / len(r1))
            
        p2 = set(ngrams(pred, 2))
        r2 = set(ngrams(ref, 2))
        if r2:
            r2_scores.append(len(p2 & r2) / len(r2))

    rouge_1_est = float(np.mean(r1_scores)) if r1_scores else 0.0
    rouge_2_est = float(np.mean(r2_scores)) if r2_scores else 0.0

    return {
        "bleu": round(float(bleu_score.score), 2),
        "rouge_1_recall": round(rouge_1_est, 4),
        "rouge_2_recall": round(rouge_2_est, 4),
        "char_limit_compliance_pct": round(compliance_rate * 100, 2),
        "official_domain_inclusion_rate": round(validity_rate * 100, 2),
        "official_domain_validity_pct": round(validity_rate * 100, 2),  # backward compatibility alias
        "link_relevance_pct": round(relevance_rate * 100, 2)
    }
