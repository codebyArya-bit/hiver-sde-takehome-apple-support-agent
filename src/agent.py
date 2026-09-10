"""
Unified AI Support Agent for @AppleSupport.
Orchestrates Intent Classification, Knowledge Base Retrieval, Calibrated Escalation Triage,
and Grounded Reply Generation into a single end-to-end pipeline.
"""

from typing import Dict, Any, Optional, List
from src.data_loader import clean_tweet_text
from src.intent_classifier import IntentClassifier
from src.retriever import AppleSupportRetriever
from src.escalation_engine import EscalationEngine
from src.generator import GroundedReplyGenerator

class AppleSupportAgent:
    """
    Production-grade AI Support Agent for @AppleSupport.
    """
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[AppleSupportRetriever] = None,
        escalation_engine: Optional[EscalationEngine] = None,
        generator: Optional[GroundedReplyGenerator] = None
    ):
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or AppleSupportRetriever()
        self.escalation_engine = escalation_engine or EscalationEngine()
        self.generator = generator or GroundedReplyGenerator()
        self._initialized = False

    def initialize(self):
        """Initializes the retriever and loads the intent classifier."""
        if not self._initialized:
            self.retriever.build_index()
            self.classifier.load()
            self._initialized = True
        return self

    def process_message(self, customer_query: str) -> Dict[str, Any]:
        """
        Executes the full agent reasoning cycle on an incoming customer inquiry.
        """
        if not self._initialized:
            self.initialize()

        cleaned_text = clean_tweet_text(customer_query)
        if not cleaned_text:
            return {
                "query": customer_query,
                "intent": "OUT_OF_SCOPE_OTHER",
                "intent_confidence": 0.0,
                "escalation_decision": "AUTO_HANDLE",
                "escalation_reason": "Empty or unintelligible message.",
                "draft_reply": "Thanks for reaching out to Apple Support! How can we help you today?",
                "grounded_in": [],
                "retrieved_evidence": []
            }

        # 1. Intent Classification
        clf_result = self.classifier.predict(cleaned_text)
        intent = clf_result["intent"]
        intent_conf = clf_result["confidence"]

        # 2. Historical Knowledge Retrieval
        retrieved_items = self.retriever.retrieve(cleaned_text, top_k=3, intent=intent)
        top_sim = retrieved_items[0]["similarity_score"] if retrieved_items else 0.0
        standard_link = self.retriever.get_standard_link(intent)

        # 3. Escalation Decision
        esc_result = self.escalation_engine.decide(
            query=cleaned_text,
            predicted_intent=intent,
            intent_confidence=intent_conf,
            top_retrieval_similarity=top_sim
        )
        escalation_decision = esc_result["decision"]
        escalation_reason = esc_result["stated_reason"]
        policy_triggered = esc_result["policy_triggered"]

        # 4. Grounded Reply Drafting with True Evidence Extraction
        gen_result = self.generator.generate_reply(
            query=cleaned_text,
            intent=intent,
            escalation_decision=escalation_decision,
            escalation_reason=escalation_reason,
            retrieved_resolutions=retrieved_items,
            standard_link=standard_link
        )

        return {
            "query": customer_query,
            "cleaned_query": cleaned_text,
            "intent": intent,
            "intent_confidence": intent_conf,
            "probabilities": clf_result.get("probabilities", {}),
            "matched_signals": clf_result.get("matched_signals", []),
            "escalation_decision": escalation_decision,
            "escalation_reason": escalation_reason,
            "policy_triggered": policy_triggered,
            "draft_reply": gen_result["draft_reply"],
            "reply_length": len(gen_result["draft_reply"]),
            "grounded_in": gen_result["grounded_in"],
            "extracted_action_used": gen_result["extracted_action_used"],
            "retrieved_evidence": retrieved_items
        }
