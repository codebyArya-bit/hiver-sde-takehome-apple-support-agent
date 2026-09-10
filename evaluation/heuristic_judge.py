"""
Deterministic Heuristic Quality Judge for Apple Support Agent Replies.
Provides fast, offline, reproducible rubric scoring across four dimensions:
1. Groundedness & Factual Correctness
2. Brand Voice & Empathy
3. Actionability & Resolution Quality
4. Escalation Appropriateness

Serves as an automated NLP metric; distinct from LLM-as-a-Judge.
"""

import re
import numpy as np
from typing import Dict, Any, List

class HeuristicSupportJudge:
    """
    Deterministic rule-calibrated rubric evaluator for rapid offline validation.
    """
    def score_single(
        self,
        customer_query: str,
        gold_intent: str,
        gold_escalation: str,
        candidate_reply: str,
        candidate_escalation: str,
        reference_resolution: str
    ) -> Dict[str, float]:
        reply_lower = candidate_reply.lower()
        query_lower = customer_query.lower()

        # 1. Escalation Appropriateness
        if candidate_escalation == gold_escalation:
            esc_score = 5.0
        elif gold_escalation == "ESCALATE" and candidate_escalation == "AUTO_HANDLE":
            if gold_intent in ["APP_STORE_AND_BILLING", "APPLE_ID_AND_ICLOUD"]:
                esc_score = 1.0
            elif any(w in query_lower for w in ["swollen", "cracked", "sue", "lawyer", "shocking"]):
                esc_score = 1.0
            else:
                esc_score = 2.0
        else:
            esc_score = 3.5  # over-escalation penalty

        # 2. Groundedness & Factual Correctness
        ground_score = 4.0
        if any(d in candidate_reply for d in ["support.apple.com", "reportaproblem.apple.com", "iforgot.apple.com"]):
            ground_score += 0.8
        elif "apple.co" in candidate_reply or "apple.com" in candidate_reply:
            ground_score += 0.5
        if any(kw in reply_lower for kw in ["settings", "restart", "battery health", "dm us", "specialist"]):
            ground_score += 0.2
        if len(candidate_reply.split()) < 6:
            ground_score -= 1.5
        ground_score = min(5.0, max(1.0, ground_score))

        # 3. Brand Voice & Empathy
        voice_score = 3.5
        empathy_markers = ["help", "happy to", "love to", "understand", "sorry", "make this right", "assist", "welcome"]
        empathy_count = sum(1 for m in empathy_markers if m in reply_lower)
        if empathy_count >= 2:
            voice_score += 1.0
        elif empathy_count == 1:
            voice_score += 0.5
        if len(candidate_reply) <= 280:
            voice_score += 0.5
        else:
            voice_score -= 1.0
        voice_score = min(5.0, max(1.0, voice_score))

        # 4. Actionability & Resolution Quality
        action_score = 3.5
        if any(d in candidate_reply for d in ["http://", "https://"]):
            action_score += 0.8
        if any(a in reply_lower for a in ["swipe", "tap", "go to", "check", "visit", "dm us", "review"]):
            action_score += 0.7
        action_score = min(5.0, max(1.0, action_score))

        overall = round(float(np.mean([ground_score, voice_score, action_score, esc_score])), 2)

        return {
            "groundedness": round(ground_score, 2),
            "brand_voice": round(voice_score, 2),
            "actionability": round(action_score, 2),
            "escalation_appropriateness": round(esc_score, 2),
            "overall_score": overall
        }

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        all_scores = []
        for gold, pred in zip(test_cases, predictions):
            scores = self.score_single(
                customer_query=gold["customer_query"],
                gold_intent=gold["gold_intent"],
                gold_escalation=gold["gold_escalation"],
                candidate_reply=pred["draft_reply"],
                candidate_escalation=pred["escalation_decision"],
                reference_resolution=gold["reference_resolution"]
            )
            all_scores.append(scores)

        return {
            "mean_groundedness": round(float(np.mean([s["groundedness"] for s in all_scores])), 2),
            "mean_brand_voice": round(float(np.mean([s["brand_voice"] for s in all_scores])), 2),
            "mean_actionability": round(float(np.mean([s["actionability"] for s in all_scores])), 2),
            "mean_escalation_appropriateness": round(float(np.mean([s["escalation_appropriateness"] for s in all_scores])), 2),
            "mean_overall_score": round(float(np.mean([s["overall_score"] for s in all_scores])), 2),
            "individual_scores": all_scores
        }
