"""
Intent Classifier for Apple Support Twitter inquiries.
Categorizes customer messages into a domain-specific 7-intent taxonomy:
1. BATTERY_AND_HARDWARE
2. IOS_SOFTWARE_UPDATE
3. APPLE_ID_AND_ICLOUD
4. APP_STORE_AND_BILLING
5. DEVICE_SETUP_AND_USAGE
6. CUSTOMER_FEEDBACK_COMPLAINT
7. OUT_OF_SCOPE_OTHER

Trained on a separate, conversation-thread disjoint training split (data/processed/apple_train_set.json)
to strictly prevent data leakage with the holdout Golden Evaluation Set.
"""

import os
import re
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

INTENTS = [
    "BATTERY_AND_HARDWARE",
    "IOS_SOFTWARE_UPDATE",
    "APPLE_ID_AND_ICLOUD",
    "APP_STORE_AND_BILLING",
    "DEVICE_SETUP_AND_USAGE",
    "CUSTOMER_FEEDBACK_COMPLAINT",
    "OUT_OF_SCOPE_OTHER"
]

DOMAIN_PATTERNS = {
    "APP_STORE_AND_BILLING": [
        r'\b(refund|charge[ds]?|subscription|receipt|billing|credit card|purchased?|app store charge|money back|itunes)\b'
    ],
    "APPLE_ID_AND_ICLOUD": [
        r'\b(apple id|icloud|locked out?|passcode|password|two[- ]factor|2fa|verification code|activation lock|compromised|reset password)\b'
    ],
    "BATTERY_AND_HARDWARE": [
        r'\b(battery|drain|draining|overheat|hot|screen cracked|broken screen|speaker|mic(rophone)?|camera|charging port|hardware|swollen|bulging|water damage)\b'
    ],
    "CUSTOMER_FEEDBACK_COMPLAINT": [
        r'\b(terrible|worst|unacceptable|sue|lawyer|garbage|furious|supervisor|manager|waste of money|fraud|incompetent|ridiculous|disgusted|hate apple|livid|trash|sucks)\b'
    ],
    "IOS_SOFTWARE_UPDATE": [
        r'\b(update|ios\s*\d+|freeze|freezing|frozen|crash|crashing|lag|bluetooth|wi-?fi|cellular|no sim|reboot|boot loop|glitch|buggy|touch screen)\b'
    ],
    "DEVICE_SETUP_AND_USAGE": [
        r'\b(how (do|can) i|how to|transfer|setup|set up|icon|padlock|orientation lock|setting|customize|feature|sync photos|airdrop)\b'
    ],
    "OUT_OF_SCOPE_OTHER": [
        r'\b(gracias|actualizaci|ayuda|hola|vuestra|merci|bonjour|privet message)\b'
    ]
}

class IntentClassifier:
    """
    Calibrated intent classifier combining n-gram TF-IDF representations
    with calibrated logistic scoring and domain pattern boosting.
    """
    def __init__(self, model_path: str = "data/processed/intent_classifier.joblib"):
        self.model_path = Path(model_path)
        self.pipeline = None
        self.intents = INTENTS
        
    def train(self, training_data: List[Dict[str, Any]]):
        """
        Trains the classifier on labeled instances from the disjoint training set.
        """
        texts = [item['customer_query'] for item in training_data]
        labels = [item['gold_intent'] for item in training_data]
        
        base_lr = LogisticRegression(
            C=1.5,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                strip_accents='unicode',
                token_pattern=r'(?u)\b\w+\b'
            )),
            ('clf', CalibratedClassifierCV(estimator=base_lr, method='sigmoid', cv=3))
        ])
        
        pipeline.fit(texts, labels)
        self.pipeline = pipeline
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.model_path)
        return self

    def load(self):
        """Loads cached model if available."""
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
        return self

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts intent with calibrated confidence and top indicative signals.
        """
        if not text or not text.strip():
            return {
                "intent": "OUT_OF_SCOPE_OTHER",
                "confidence": 0.5,
                "probabilities": {k: 0.0 for k in self.intents},
                "matched_signals": []
            }

        text_lower = text.lower()
        matched_signals = []
        rule_scores = {intent: 0.0 for intent in self.intents}

        for intent, patterns in DOMAIN_PATTERNS.items():
            for pat in patterns:
                matches = re.findall(pat, text_lower)
                if matches:
                    matched_signals.extend([m[0] if isinstance(m, tuple) else m for m in matches])
                    # Higher weight for safety-critical domains
                    weight = 4.0 if intent in ["APP_STORE_AND_BILLING", "APPLE_ID_AND_ICLOUD"] else 3.0
                    rule_scores[intent] += (len(matches) * weight)

        # Predict with statistical pipeline
        if self.pipeline is None:
            if self.model_path.exists():
                self.load()

        if self.pipeline is not None:
            probs = self.pipeline.predict_proba([text])[0]
            classes = self.pipeline.classes_
            prob_dict = {cls_name: float(p) for cls_name, p in zip(classes, probs)}
            for missing_cls in self.intents:
                if missing_cls not in prob_dict:
                    prob_dict[missing_cls] = 0.001
        else:
            prob_dict = {k: 1.0 / len(self.intents) for k in self.intents}

        # Bayesian fusion of statistical probabilities and explicit domain signals
        for intent, score in rule_scores.items():
            if score > 0:
                prob_dict[intent] += (score * 0.45)

        total = sum(prob_dict.values())
        if total > 0:
            prob_dict = {k: v / total for k, v in prob_dict.items()}

        best_intent = max(prob_dict.keys(), key=lambda k: prob_dict[k])
        confidence = float(prob_dict[best_intent])

        return {
            "intent": best_intent,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in sorted(prob_dict.items(), key=lambda x: -x[1])},
            "matched_signals": list(set(matched_signals))[:5]
        }
