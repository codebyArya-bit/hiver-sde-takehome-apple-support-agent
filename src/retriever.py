"""
Hybrid Retriever for Apple Support Knowledge Base.
Indexes historical @AppleSupport conversations and retrieves relevant historical
resolutions, standard troubleshooting instructions, and verified support URLs.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import clean_tweet_text, extract_urls

# Standard Apple Support official knowledge base URLs mapping
STANDARD_APPLE_LINKS = {
    "BATTERY_AND_HARDWARE": "https://support.apple.com/iphone/repair/battery-power",
    "IOS_SOFTWARE_UPDATE": "https://support.apple.com/HT204204",
    "APPLE_ID_AND_ICLOUD": "https://iforgot.apple.com",
    "APP_STORE_AND_BILLING": "https://reportaproblem.apple.com",
    "DEVICE_SETUP_AND_USAGE": "https://support.apple.com/guide/iphone",
    "CUSTOMER_FEEDBACK_COMPLAINT": "https://www.apple.com/feedback",
    "OUT_OF_SCOPE_OTHER": "https://support.apple.com"
}

class AppleSupportRetriever:
    """
    Retrieval engine over historical @AppleSupport conversation pairs.
    Uses TF-IDF semantic vector space to rank historical resolutions.
    """
    def __init__(self, kb_path: str = "data/processed/apple_support_kb.json"):
        self.kb_path = Path(kb_path)
        self.kb_items: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors = None
        self._is_indexed = False

    def build_index(self):
        """Builds TF-IDF index over historical queries and resolutions."""
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {self.kb_path}")
            
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            self.kb_items = json.load(f)

        corpus = [
            f"{item['customer_query']} {item['support_reply']}"
            for item in self.kb_items
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        self._is_indexed = True
        return self

    def retrieve(self, query: str, top_k: int = 3, intent: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top_k historical resolutions matching customer inquiry.
        """
        if not self._is_indexed:
            self.build_index()

        cleaned_query = clean_tweet_text(query)
        if not cleaned_query:
            return []

        q_vec = self.vectorizer.transform([cleaned_query])
        sims = cosine_similarity(q_vec, self.doc_vectors)[0]

        top_indices = np.argsort(-sims)[:top_k]
        results = []

        for idx in top_indices:
            score = float(sims[idx])
            item = self.kb_items[idx]
            
            # Extract URLs or fallback to standard official link for intent
            urls = item.get('support_urls', [])
            if not urls and intent and intent in STANDARD_APPLE_LINKS:
                urls = [STANDARD_APPLE_LINKS[intent]]
                
            results.append({
                "conversation_id": item.get('id', ''),
                "historical_query": item.get('customer_query', ''),
                "historical_reply": item.get('support_reply', ''),
                "similarity_score": round(score, 4),
                "support_urls": urls
            })

        return results

    def get_standard_link(self, intent: str) -> str:
        """Returns standard official Apple Support URL for an intent."""
        return STANDARD_APPLE_LINKS.get(intent, STANDARD_APPLE_LINKS["OUT_OF_SCOPE_OTHER"])
