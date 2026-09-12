"""
Data loading and preprocessing utilities for Customer Support Twitter conversations.
Focuses on @AppleSupport multi-turn conversations and inquiry-response pairs.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

def clean_tweet_text(text: str) -> str:
    """
    Cleans raw customer/support tweet text:
    - Normalizes user anonymization tokens (@115712, @AppleSupport)
    - Fixes encoding artifacts (e.g., smart quotes, non-breaking spaces)
    - Standardizes whitespace
    - Preserves URLs or standardizes them for link extraction
    """
    if not text:
        return ""
    
    import html
    text = html.unescape(text)
    # Replace non-breaking spaces and common Unicode artifacts
    text = text.replace('\xa0', ' ').replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2014', ' - ')
    
    # Remove leading/trailing quotes often present in csv extracts
    text = text.strip(' "')
    
    # Normalize handles like @AppleSupport, @123456 -> @handle or remove anonymized ID tags
    text = re.sub(r'@\d+', '', text)  # remove numeric user IDs
    text = re.sub(r'@AppleSupport\b', '', text, flags=re.IGNORECASE)
    
    # Compress multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_urls(text: str) -> List[str]:
    """Extracts all URLs present in a tweet."""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    return re.findall(url_pattern, text)

def load_conversation_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Loads preprocessed conversation dataset from JSON."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_pair(pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Cleans and validates a customer-support conversation pair.
    Returns None if the pair is degenerate or too short to be useful.
    """
    query = clean_tweet_text(pair.get('customer_query', ''))
    reply = clean_tweet_text(pair.get('support_reply', ''))
    
    # Filter out empty or trivially short queries (e.g. "?", "hello", "hi")
    if len(query.split()) < 3 or len(reply.split()) < 3:
        return None
        
    return {
        'conversation_id': pair.get('conversation_id', ''),
        'customer_query': query,
        'support_reply': reply,
        'reply_urls': extract_urls(pair.get('support_reply', '')),
        'raw_customer': pair.get('customer_query', ''),
        'raw_support': pair.get('support_reply', '')
    }
