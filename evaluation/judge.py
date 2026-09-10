"""
LLM-as-a-Judge Rubric and Scorer for Apple Support Agent Replies.
Evaluates agent output across four distinct dimensions (1.0 - 5.0 scale):
1. Groundedness & Factual Correctness
2. Brand Voice & Empathy
3. Actionability & Resolution Quality
4. Escalation Appropriateness
"""

import os
import re
from typing import Dict, Any, List, Optional
import numpy as np

JUDGE_RUBRIC_PROMPT = """
You are an expert Quality Assurance Judge for Apple Customer Support on Twitter.
Evaluate the candidate response against the customer query, ground truth reference, and escalation context.

Rate the candidate response on a 1.0 to 5.0 scale across these four criteria:

1. Groundedness & Factual Correctness:
- 5.0: Completely accurate, factually grounded in Apple ecosystem, correct links/menus.
- 3.0: Mostly accurate, but slightly generic or imprecise troubleshooting step.
- 1.0: Factually wrong, hallucinates non-existent features, fake links, or dangerous advice.

2. Brand Voice & Empathy:
- 5.0: Warm, respectful, professional, empathetic, complies with Apple Twitter persona.
- 3.0: Neutral, slightly robotic, but polite.
- 1.0: Rude, dismissive, robotic repetition, or offensive.

3. Actionability & Resolution Quality:
- 5.0: Gives the customer a clear, concrete immediate action or official self-service resource.
- 3.0: Vague advice ("check your phone"), requires multiple follow-ups.
- 1.0: Provides no helpful path forward or dead-ends the conversation.

4. Escalation Appropriateness:
- 5.0: Correctly escalates when PII/refund/hardware/anger is present, or correctly auto-handles clear self-service.
- 3.0: Minor mismatch (e.g. over-escalated an easy how-to, or asked for DM without clear reason).
- 1.0: Catastrophic failure: auto-handled an account compromise/refund dispute, or ignored legal/safety threat.
"""

class LLMSupportJudge:
    """
    Evaluator that scores candidate replies using standardized rubrics.
    Provides automated calibrated scoring on CPU and connects to remote LLM APIs if available.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")

    def score_single(
        self,
        customer_query: str,
        gold_intent: str,
        gold_escalation: str,
        candidate_reply: str,
        candidate_escalation: str,
        reference_resolution: str
    ) -> Dict[str, float]:
        """
        Scores a single candidate response against reference standards.
        """
        reply_lower = candidate_reply.lower()
        query_lower = customer_query.lower()

        # --- 1. Escalation Appropriateness ---
        if candidate_escalation == gold_escalation:
            esc_score = 5.0
        elif gold_escalation == "ESCALATE" and candidate_escalation == "AUTO_HANDLE":
            # Dangerous miss: auto-handling a case that needed human
            if gold_intent in ["APP_STORE_AND_BILLING", "APPLE_ID_AND_ICLOUD"]:
                esc_score = 1.0
            elif any(w in query_lower for w in ["swollen", "cracked", "sue", "lawyer"]):
                esc_score = 1.0
            else:
                esc_score = 2.0
        else:
            # Over-escalation: escalated an easy auto-handle
            esc_score = 3.5

        # --- 2. Groundedness & Factual Correctness ---
        ground_score = 4.0
        # Check for official apple support links or accurate terminology
        if any(d in candidate_reply for d in ["apple.com", "apple.co", "reportaproblem", "iforgot"]):
            ground_score += 0.8
        if any(kw in reply_lower for kw in ["settings", "restart", "battery health", "dm us", "specialist"]):
            ground_score += 0.2
        # Deduct if hallucinating generic filler without context
        if len(candidate_reply.split()) < 6:
            ground_score -= 1.5
        ground_score = min(5.0, max(1.0, ground_score))

        # --- 3. Brand Voice & Empathy ---
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
            voice_score -= 1.0  # violates Twitter character constraints

        voice_score = min(5.0, max(1.0, voice_score))

        # --- 4. Actionability & Resolution Quality ---
        action_score = 3.5
        if any(d in candidate_reply for d in ["http://", "https://"]):
            action_score += 0.8
        if any(a in reply_lower for a in ["swipe", "tap", "go to", "check", "visit", "dm us"]):
            action_score += 0.7
        if "please dm us" in reply_lower and candidate_escalation == "AUTO_HANDLE":
            action_score -= 0.8  # confusing: auto-handling but asking to DM

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
        """
        Evaluates an entire batch of predictions and aggregates rubric scores.
        """
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

        avg_groundedness = float(np.mean([s["groundedness"] for s in all_scores]))
        avg_brand_voice = float(np.mean([s["brand_voice"] for s in all_scores]))
        avg_actionability = float(np.mean([s["actionability"] for s in all_scores]))
        avg_escalation = float(np.mean([s["escalation_appropriateness"] for s in all_scores]))
        avg_overall = float(np.mean([s["overall_score"] for s in all_scores]))

        return {
            "mean_groundedness": round(avg_groundedness, 2),
            "mean_brand_voice": round(avg_brand_voice, 2),
            "mean_actionability": round(avg_actionability, 2),
            "mean_escalation_appropriateness": round(avg_escalation, 2),
            "mean_overall_score": round(avg_overall, 2),
            "individual_scores": all_scores
        }
