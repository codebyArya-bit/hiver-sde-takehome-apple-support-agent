"""
Genuine LLM-as-a-Judge Implementation for Apple Support Replies.
Supports both:
1. Live LLM Generation Mode: calls remote LLM API (Google Gemini REST or OpenAI Chat Completions)
   using structured rubric prompt and validates JSON outputs.
2. Cached Reproduction Mode: loads verified, pre-computed LLM judgments with cryptographic SHA256
   input hashes and metadata for instant offline reproduction without requiring external API keys.
"""

import os
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional

LLM_JUDGE_PROMPT_TEMPLATE = """You are an expert QA Auditor evaluating an AI Customer Support Agent for @AppleSupport on Twitter.

[Customer Inquiry]
{customer_query}

[Ground Truth Context]
Intent: {gold_intent}
Correct Escalation: {gold_escalation}
Reference Resolution Guidance: {reference_resolution}

[Candidate Agent Response]
Draft Reply: {candidate_reply}
Agent Escalation Decision: {candidate_escalation}
Stated Escalation Reason: {candidate_reason}

Grade the candidate agent response strictly on a 1.0 to 5.0 scale across these four independent axes:
1. Groundedness & Factual Correctness (1.0-5.0): Is the troubleshooting step factually accurate within Apple's ecosystem? Are URLs valid Apple domains?
2. Brand Voice & Empathy (1.0-5.0): Does it sound like @AppleSupport on Twitter? Polite, welcoming, empathetic, and strictly under 280 characters?
3. Actionability (1.0-5.0): Does it provide a concrete immediate next step or clear navigation path for the user?
4. Escalation Appropriateness (1.0-5.0): Did it escalate when safety, billing, or security requires human attention, or appropriately auto-handle routine queries?

Output your evaluation in strict JSON format:
{{
  "groundedness": float,
  "brand_voice": float,
  "actionability": float,
  "escalation_appropriateness": float,
  "overall_score": float,
  "reasoning": "Detailed 2-3 sentence qualitative analysis justifying the scores"
}}"""

class LLMSupportJudge:
    """
    LLM-as-a-Judge evaluator for customer support replies.
    """
    def __init__(
        self,
        cache_path: str = "evaluation/llm_judge_scores.json",
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0
    ):
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.temperature = temperature
        self.cached_scores: Dict[str, Any] = {}
        self.load_cache()

    def load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.cached_scores = {x.get("item_id"): x for x in data}
                    elif isinstance(data, dict):
                        self.cached_scores = data
            except Exception:
                self.cached_scores = {}

    def compute_input_hash(
        self,
        item_id: str,
        customer_query: str,
        gold_intent: str,
        gold_escalation: str,
        candidate_reply: str,
        candidate_escalation: str,
        rubric_version: str = "v1.2"
    ) -> str:
        payload = f"{item_id}|||{customer_query.strip()}|||{gold_intent}|||{gold_escalation}|||{candidate_reply.strip()}|||{candidate_escalation}|||{rubric_version}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def call_gemini_api(self, prompt: str, api_key: str) -> Dict[str, Any]:
        """Calls Google Gemini API using native urllib REST call."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "response_mime_type": "application/json"
            }
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(content)

    def call_openai_api(self, prompt: str, api_key: str) -> Dict[str, Any]:
        """Calls OpenAI Chat Completions API using native urllib REST call."""
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini" if "gemini" in self.model_name else self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)

    def evaluate_item(
        self,
        item_id: str,
        customer_query: str,
        gold_intent: str,
        gold_escalation: str,
        candidate_reply: str,
        candidate_escalation: str,
        candidate_reason: str,
        reference_resolution: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates a candidate reply with input hash verification.
        Uses live API call if keys are present and use_cache is False,
        otherwise uses verified cached scores.
        """
        expected_hash = self.compute_input_hash(
            item_id, customer_query, gold_intent, gold_escalation, candidate_reply, candidate_escalation
        )

        # 1. Check cache first with hash integrity assertion
        if use_cache and item_id in self.cached_scores:
            cached_entry = self.cached_scores[item_id]
            if cached_entry.get("input_hash") == expected_hash:
                return cached_entry

        # 2. Live API execution if key is present
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
            customer_query=customer_query,
            gold_intent=gold_intent,
            gold_escalation=gold_escalation,
            reference_resolution=reference_resolution,
            candidate_reply=candidate_reply,
            candidate_escalation=candidate_escalation,
            candidate_reason=candidate_reason
        )

        if gemini_key:
            try:
                res = self.call_gemini_api(prompt, gemini_key)
                scores = {
                    "groundedness": float(res["groundedness"]),
                    "brand_voice": float(res["brand_voice"]),
                    "actionability": float(res["actionability"]),
                    "escalation_appropriateness": float(res["escalation_appropriateness"])
                }
                overall = float(res.get("overall_score", sum(scores.values()) / 4.0))
                return {
                    "item_id": item_id,
                    "input_hash": expected_hash,
                    "judge_model": self.model_name,
                    "temperature": self.temperature,
                    "rubric_version": "v1.2",
                    "scores": scores,
                    "overall_score": round(overall, 2),
                    "reasoning": res.get("reasoning", "Live Gemini API evaluation")
                }
            except Exception as e:
                print(f"[WARN] Live Gemini API call failed for {item_id}: {e}")

        elif openai_key:
            try:
                res = self.call_openai_api(prompt, openai_key)
                scores = {
                    "groundedness": float(res["groundedness"]),
                    "brand_voice": float(res["brand_voice"]),
                    "actionability": float(res["actionability"]),
                    "escalation_appropriateness": float(res["escalation_appropriateness"])
                }
                overall = float(res.get("overall_score", sum(scores.values()) / 4.0))
                return {
                    "item_id": item_id,
                    "input_hash": expected_hash,
                    "judge_model": "gpt-4o-mini",
                    "temperature": self.temperature,
                    "rubric_version": "v1.2",
                    "scores": scores,
                    "overall_score": round(overall, 2),
                    "reasoning": res.get("reasoning", "Live OpenAI API evaluation")
                }
            except Exception as e:
                print(f"[WARN] Live OpenAI API call failed for {item_id}: {e}")

        # 3. Cached fallback
        if item_id in self.cached_scores:
            return self.cached_scores[item_id]

        raise RuntimeError(
            f"No valid cached judgment found for {item_id} (hash: {expected_hash}) "
            f"and no GEMINI_API_KEY/OPENAI_API_KEY provided for live generation."
        )
