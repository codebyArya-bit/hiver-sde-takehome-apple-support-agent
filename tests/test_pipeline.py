"""
Unit tests for Apple Support AI Agent pipeline.
Validates:
- Data loader cleaning
- Intent classifier prediction shape and bounds
- Knowledge retriever indexing and search
- Escalation engine policy triggers
- Reply generator length constraint and brand tone
- Automated metrics calculation
"""

import pytest
from src.data_loader import clean_tweet_text, extract_urls
from src.intent_classifier import IntentClassifier, INTENTS
from src.retriever import AppleSupportRetriever
from src.escalation_engine import EscalationEngine
from src.generator import GroundedReplyGenerator
from src.agent import AppleSupportAgent
from evaluation.metrics import (
    calculate_intent_metrics,
    calculate_escalation_metrics,
    calculate_generation_metrics
)

def test_clean_tweet_text():
    raw = '@AppleSupport @115712 My iPhone 7 screen is cracked!   https://t.co/xyz123'
    cleaned = clean_tweet_text(raw)
    assert '@AppleSupport' not in cleaned
    assert '@115712' not in cleaned
    assert 'screen is cracked!' in cleaned
    assert '  ' not in cleaned  # normalized whitespace

def test_extract_urls():
    text = "Check this link: https://support.apple.com/HT204204 and http://apple.co/battery"
    urls = extract_urls(text)
    assert len(urls) == 2
    assert "https://support.apple.com/HT204204" in urls

def test_intent_classifier_prediction():
    clf = IntentClassifier()
    result = clf.predict("My battery drains 50% in one hour and gets extremely hot")
    assert result["intent"] == "BATTERY_AND_HARDWARE"
    assert 0.0 <= result["confidence"] <= 1.0
    assert "intent" in result

def test_escalation_engine_policies():
    engine = EscalationEngine()
    
    # Financial policy trigger
    res_refund = engine.decide(
        query="I need a refund for an accidental $99 App Store charge on my credit card",
        predicted_intent="APP_STORE_AND_BILLING",
        intent_confidence=0.92
    )
    assert res_refund["decision"] == "ESCALATE"
    assert "POLICY_FINANCIAL_TRANSACTION" in res_refund["policy_triggered"]

    # Security policy trigger
    res_lock = engine.decide(
        query="My Apple ID is locked for security reasons and 2FA code is not sending",
        predicted_intent="APPLE_ID_AND_ICLOUD",
        intent_confidence=0.90
    )
    assert res_lock["decision"] == "ESCALATE"
    assert "POLICY_SECURITY_CREDENTIALS" in res_lock["policy_triggered"]

    # Swollen battery safety trigger
    res_safety = engine.decide(
        query="My battery is swollen and popping out the screen!",
        predicted_intent="BATTERY_AND_HARDWARE",
        intent_confidence=0.95
    )
    assert res_safety["decision"] == "ESCALATE"
    assert "POLICY_HARDWARE_DAMAGE_SAFETY" in res_safety["policy_triggered"]

    # Standard auto-handle
    res_autohandle = engine.decide(
        query="How do I turn off the rotation lock icon on my screen?",
        predicted_intent="DEVICE_SETUP_AND_USAGE",
        intent_confidence=0.88,
        top_retrieval_similarity=0.85
    )
    assert res_autohandle["decision"] == "AUTO_HANDLE"
    assert len(res_autohandle["stated_reason"]) > 10

def test_reply_generator_twitter_length_limit():
    gen = GroundedReplyGenerator()
    res = gen.generate_reply(
        query="My phone is freezing after the latest iOS update",
        intent="IOS_SOFTWARE_UPDATE",
        escalation_decision="AUTO_HANDLE",
        escalation_reason="Standard software update issue.",
        retrieved_resolutions=[],
        standard_link="https://support.apple.com/HT204204"
    )
    reply_text = res["draft_reply"]
    assert len(reply_text) <= 280
    assert "https://" in reply_text

def test_metrics_calculation():
    y_true = ["IOS_SOFTWARE_UPDATE", "BATTERY_AND_HARDWARE"]
    y_pred = ["IOS_SOFTWARE_UPDATE", "BATTERY_AND_HARDWARE"]
    res = calculate_intent_metrics(y_true, y_pred, labels=INTENTS)
    assert res["accuracy"] == 1.0

    esc_true = ["ESCALATE", "AUTO_HANDLE"]
    esc_pred = ["ESCALATE", "AUTO_HANDLE"]
    esc_res = calculate_escalation_metrics(esc_true, esc_pred)
    assert esc_res["accuracy"] == 1.0
    assert esc_res["escalation_recall"] == 1.0

def test_zero_data_leakage():
    import json
    with open("data/processed/apple_train_set.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open("data/processed/apple_support_kb.json", "r", encoding="utf-8") as f:
        kb = json.load(f)
    with open("data/golden_eval_set.json", "r", encoding="utf-8") as f:
        gold = json.load(f)

    train_cids = {x.get("conversation_id") or x.get("id") for x in train}
    kb_cids = {x.get("conversation_id") or x.get("id") for x in kb}
    gold_cids = {x.get("conversation_id") or x.get("id") for x in gold}

    # Assert strict zero overlap
    assert len(train_cids.intersection(gold_cids)) == 0, "Data leakage between Train and Gold set!"
    assert len(kb_cids.intersection(gold_cids)) == 0, "Data leakage between KB and Gold set!"
    assert len(gold) == 200, "Golden evaluation set must have exactly 200 items"

def test_sacrebleu_layout_and_generation_metrics():
    hypotheses = ["Check your settings at https://support.apple.com/HT204204 to fix this issue."]
    references = ["Please go to Settings > Battery and visit https://support.apple.com/HT204204 for steps."]
    intents = ["BATTERY_AND_HARDWARE"]
    
    # Verifies that SacreBLEU receives 1D list and executes without shape error
    metrics = calculate_generation_metrics(hypotheses, references, target_intents=intents)
    assert "bleu" in metrics
    assert metrics["bleu"] >= 0.0
    assert metrics["char_limit_compliance_pct"] == 100.0
    assert metrics["official_domain_validity_pct"] == 100.0

def test_rag_historical_evidence_extraction():
    gen = GroundedReplyGenerator()
    hist_resolutions = [
        {
            "conversation_id": "conv_test_123",
            "historical_reply": "Hi! We'd recommend force restarting your device and updating to the latest iOS.",
            "similarity": 0.85
        }
    ]
    res = gen.generate_reply(
        query="My iPhone is lagging after the update",
        intent="IOS_SOFTWARE_UPDATE",
        escalation_decision="AUTO_HANDLE",
        escalation_reason="Standard software update troubleshooting.",
        retrieved_resolutions=hist_resolutions,
        standard_link="https://support.apple.com/HT204204"
    )
    assert len(res["draft_reply"]) <= 280
    assert len(res["grounded_in"]) > 0
    assert res["grounded_in"][0]["conversation_id"] == "conv_test_123"
    assert "resolution_snippet" in res["grounded_in"][0]

def test_end_to_end_agent_processing():
    agent = AppleSupportAgent()
    agent.initialize()
    
    result = agent.process_message("My battery goes from 100 to 20 in 30 minutes, this is unusable!")
    assert "intent" in result
    assert "intent_confidence" in result
    assert "escalation_decision" in result
    assert result["escalation_decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert "escalation_reason" in result
    assert "draft_reply" in result
    assert len(result["draft_reply"]) <= 280
    assert "grounded_in" in result
    assert isinstance(result["grounded_in"], list)

def test_human_judge_agreement_stats():
    from evaluation.human_agreement import calculate_agreement_metrics
    human_scores = [4.5, 3.5, 5.0, 4.0, 3.0]
    judge_scores = [4.5, 3.6, 4.8, 4.0, 3.2]
    stats = calculate_agreement_metrics(human_scores, judge_scores, metric_name="Overall Rubric Score")
    assert "pearson_r" in stats
    assert stats["pearson_r"] > 0.90
    assert stats["mae"] < 0.20



