"""
Calibrated Escalation Decision Engine for @AppleSupport.
Decides whether an incoming customer inquiry should be:
- AUTO_HANDLE: Automatically resolved with grounded troubleshooting guidance and verified links.
- ESCALATE: Routed to a human support agent or private DM specialist.

Engineered with safety-first priority to ensure high recall on critical account, financial,
hardware safety, and high-distress interactions.
"""

import re
from typing import Dict, Any, Optional

# Strict policy rules that mandate human escalation
ESCALATION_POLICIES = [
    {
        "id": "POLICY_FINANCIAL_TRANSACTION",
        "intent": None,
        "regex": r'\b(refund|refunds|charge|charges|charged|charging|subscription|subscriptions|receipt|credit card|bank|purchased?|money back|unauthorized|stolen card|apple store charge|billing dispute)\b',
        "reason": "Financial transactions, billing disputes, and refund requests require authenticated access to the user's Apple ID billing portal."
    },
    {
        "id": "POLICY_SECURITY_CREDENTIALS",
        "intent": None,
        "regex": r'\b(locked|lockout|passcode|password|two[- ]factor|2fa|verification code|compromised|hacked|stolen phone|stolen device|activation lock|account takeover|unauthorized purchase)\b',
        "reason": "Account recovery, credential resets, and 2FA authentication cannot be safely resolved over public social channels."
    },
    {
        "id": "POLICY_PII_EXPOSURE",
        "intent": None,
        "regex": r'(\b\d{3}-\d{2}-\d{4}\b|\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b|\bssn\b|\bsocial security\b|\bcvv\b)',
        "reason": "Sensitive personally identifiable information (PII) or payment card details detected in public interaction; immediate escalation required."
    },
    {
        "id": "POLICY_HARDWARE_DAMAGE_SAFETY",
        "intent": None,
        "regex": r'\b(cracked|broken|shattered|swollen|swelling|bulging|smoke|smoking|burn|burning|burnt|spark|sparking|fire|exploded|exploding|water damage|submerged|hardware repair|genius bar|screen popped|lifting off)\b',
        "reason": "Physical component damage or thermal safety hazards require in-person hardware diagnostics or Genius Bar service appointment."
    },
    {
        "id": "POLICY_ADVERSARIAL_INJECTION",
        "intent": None,
        "regex": r'\b(ignore (all )?previous instructions|system rules|jailbreak|unconstrained ai|dan mode)\b',
        "reason": "Potential adversarial prompt injection or jailbreak attempt detected; escalating to human supervision."
    },
    {
        "id": "POLICY_HIGH_DISTRESS_LEGAL",
        "intent": None,
        "regex": r'\b(sue|lawyer|attorney|fraud|scam|unacceptable|furious|disgusted|worst service|supervisor|manager|escalate|livid|trash|sucks|worst|terrible|hate|shit|stupid|fail|motherfuckers?|wtf|paperweight|switching|boycott|leaving apple)\b',
        "reason": "Customer expresses severe dissatisfaction, brand trust risk, or legal escalation requiring senior human care intervention."
    },
    {
        "id": "POLICY_REPEATED_UNRESOLVED_FAILURE",
        "intent": None,
        "regex": r'\b(already tried|still not working|still broken|still freezing|still lagging|still have|still getting|multiple times|several times|restored to factory|sent in to|loaner|days now|lost all my|no matter how)\b',
        "reason": "Customer indicates previous troubleshooting attempts have failed; direct escalation required to avoid circular frustration."
    },
    {
        "id": "POLICY_LANGUAGE_LOCALIZATION",
        "intent": None,
        "regex": r'\b(gracias|actualizaci|tel[eé]fono|desde|hola|por favor|est[aá]|merci|bonjour|c\'est|que hago|como puedo)\b',
        "reason": "Non-English message requires routing to a native-language support queue."
    },
    {
        "id": "POLICY_PRIVATE_MESSAGE_TRANSFER",
        "intent": None,
        "regex": r'\b(privet message|private message|check dm|inbox|sent a dm|sent a message)\b',
        "reason": "Customer already initiated private conversation; routing directly to Twitter DM specialist queue."
    }
]

class EscalationEngine:
    """
    Evaluates customer intent, confidence scores, retrieval grounding, and policy triggers
    to render an authoritative AUTO_HANDLE vs. ESCALATE decision.
    """
    def __init__(self, min_intent_confidence: float = 0.40, min_retrieval_similarity: float = 0.05):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

    def decide(
        self,
        query: str,
        predicted_intent: str,
        intent_confidence: float,
        top_retrieval_similarity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Renders an escalation triage decision with explicit reasoning.
        """
        query_lower = query.lower()

        # 1. Check Hard Escalation Policies
        for policy in ESCALATION_POLICIES:
            intent_matches = (policy["intent"] is None) or (policy["intent"] == predicted_intent)
            pattern_matches = bool(re.search(policy["regex"], query_lower))

            if intent_matches and pattern_matches:
                return {
                    "decision": "ESCALATE",
                    "stated_reason": policy["reason"],
                    "policy_triggered": policy["id"],
                    "confidence": 0.95
                }

        # 2. Intent-specific default escalation rules
        if predicted_intent in ["APP_STORE_AND_BILLING", "APPLE_ID_AND_ICLOUD"]:
            return {
                "decision": "ESCALATE",
                "stated_reason": f"{predicted_intent.replace('_', ' ').title()} inquiries involve sensitive PII and account security.",
                "policy_triggered": "POLICY_SENSITIVE_DOMAIN",
                "confidence": 0.90
            }

        if predicted_intent == "CUSTOMER_FEEDBACK_COMPLAINT":
            return {
                "decision": "ESCALATE",
                "stated_reason": "Brand feedback and complaints require empathetic human engagement to mitigate churn.",
                "policy_triggered": "POLICY_CUSTOMER_COMPLAINT",
                "confidence": 0.88
            }

        # 3. Model Uncertainty & Low Confidence Safety Catch
        if intent_confidence < self.min_intent_confidence:
            return {
                "decision": "ESCALATE",
                "stated_reason": f"Model intent confidence is low ({intent_confidence:.2f} < {self.min_intent_confidence:.2f}); routing to human to prevent incorrect automated guidance.",
                "policy_triggered": "POLICY_LOW_MODEL_CONFIDENCE",
                "confidence": round(1.0 - intent_confidence, 2)
            }

        # 4. Knowledge Retrieval Grounding Deficiency Catch
        if top_retrieval_similarity < self.min_retrieval_similarity:
            return {
                "decision": "ESCALATE",
                "stated_reason": f"Historical resolution similarity is low ({top_retrieval_similarity:.2f} < {self.min_retrieval_similarity:.2f}); no confident verified troubleshooting precedent found.",
                "policy_triggered": "POLICY_INSUFFICIENT_RETRIEVAL_GROUNDING",
                "confidence": 0.85
            }

        # 5. Safe for Auto-Handling
        auto_reasons = {
            "BATTERY_AND_HARDWARE": "Standard on-device battery health diagnostic and power-saving optimization steps available.",
            "IOS_SOFTWARE_UPDATE": "Standard iOS software troubleshooting steps (restart, storage check, supplemental update) apply.",
            "DEVICE_SETUP_AND_USAGE": "Deterministic UI navigation and feature configuration instructions available.",
            "OUT_OF_SCOPE_OTHER": "General product inquiry addressable with standard public resources."
        }

        reason = auto_reasons.get(
            predicted_intent,
            "Standard self-service support query with high model confidence and verified historical precedent."
        )

        return {
            "decision": "AUTO_HANDLE",
            "stated_reason": reason,
            "policy_triggered": "NONE_AUTO_HANDLED",
            "confidence": round(intent_confidence, 2)
        }
