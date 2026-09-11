"""
Simple Baseline Agent for Apple Support.
Represents a traditional non-LLM / standard statistical ML baseline:
- Intent Classification: Standard Multinomial Naive Bayes with CountVectorizer
- Escalation Decision: Simple keyword-matching list
- Reply Generation: 1-Nearest-Neighbor raw historical reply retrieval
"""

import re
from typing import Dict, Any, List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from src.data_loader import clean_tweet_text
from src.retriever import AppleSupportRetriever

SIMPLE_ESCALATION_KEYWORDS = [
    r'\brefund\b', r'\bhuman\b', r'\bagent\b', r'\bmanager\b',
    r'\blawyer\b', r'\bsue\b', r'\bstolen\b', r'\bhacked\b'
]

class SimpleBaselineAgent:
    """
    Standard ML baseline: Naive Bayes + Keyword matching + 1-NN raw retrieval.
    """
    def __init__(self, retriever: AppleSupportRetriever):
        self.retriever = retriever
        self.pipeline = None

    def train_classifier(self, training_data: List[Dict[str, Any]]):
        texts = [item['customer_query'] for item in training_data]
        labels = [item['gold_intent'] for item in training_data]
        self.pipeline = Pipeline([
            ('vec', CountVectorizer(ngram_range=(1, 1))),
            ('nb', MultinomialNB())
        ])
        self.pipeline.fit(texts, labels)
        return self

    def process_message(
        self,
        customer_query: str,
        context_history: Any = None
    ) -> Dict[str, Any]:
        cleaned = clean_tweet_text(customer_query)
        
        # 1. Intent via simple Naive Bayes
        prob_dict = {}
        if self.pipeline:
            pred_intent = self.pipeline.predict([cleaned])[0]
            probs = self.pipeline.predict_proba([cleaned])[0]
            classes = self.pipeline.classes_
            prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
            conf = float(max(probs))
        else:
            pred_intent = "IOS_SOFTWARE_UPDATE"
            conf = 0.50

        # 2. Simple Keyword Escalation
        escalate = False
        query_lower = cleaned.lower()
        for kw in SIMPLE_ESCALATION_KEYWORDS:
            if re.search(kw, query_lower):
                escalate = True
                break

        escalation_decision = "ESCALATE" if escalate else "AUTO_HANDLE"
        escalation_reason = "Simple keyword match trigger." if escalate else "No negative keywords found."

        # 3. 1-NN Raw Historical Reply
        retrieved = self.retriever.retrieve(cleaned, top_k=1, intent=pred_intent)
        raw_reply = retrieved[0]["historical_reply"] if retrieved else "Please contact Apple Support for assistance."

        return {
            "query": customer_query,
            "intent": pred_intent,
            "intent_confidence": round(conf, 4),
            "intent_probabilities": prob_dict,
            "escalation_decision": escalation_decision,
            "escalation_reason": escalation_reason,
            "draft_reply": raw_reply,
            "reply_length": len(raw_reply),
            "retrieved_evidence": retrieved
        }
