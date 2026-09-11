"""
Trivial Baseline for Apple Support Agent.
Represents the naive zero-intelligence baseline:
- Predicts majority intent class: IOS_SOFTWARE_UPDATE
- Always predicts AUTO_HANDLE (the majority triage class, 71% of incoming traffic)
- Emits a static generic canned macro reply

Demonstrates why high raw accuracy (71%) is deceptive when minority-class recall is 0.0%.
"""

from typing import Dict, Any

class TrivialBaselineAgent:
    """
    Trivial baseline predicting majority classes and emitting a static macro reply.
    Deterministic triage decision: Always predict AUTO_HANDLE.
    """
    def __init__(self):
        self.majority_intent = "IOS_SOFTWARE_UPDATE"
        self.majority_escalation = "AUTO_HANDLE"
        self.static_macro_reply = "Thanks for reaching out! We'd love to help. Please DM us your device model and iOS version so we can look into this."

    def process_message(
        self,
        customer_query: str,
        context_history: Any = None
    ) -> Dict[str, Any]:
        return {
            "query": customer_query,
            "intent": self.majority_intent,
            "intent_confidence": 0.39,  # baseline majority class prior
            "escalation_decision": self.majority_escalation,
            "escalation_reason": "Trivial baseline default: always predict AUTO_HANDLE (majority class).",
            "draft_reply": self.static_macro_reply,
            "reply_length": len(self.static_macro_reply),
            "retrieved_evidence": []
        }
