"""
Stratified Sampling and Freezing of N=50 Evaluation Benchmark.
Selects 50 items from data/golden_eval_set.json stratified by (gold_intent, gold_escalation, difficulty)
with fixed random_state=42, executes current Proposed AI Agent, and stamps each record with a SHA256 input hash.
"""

import sys
import json
import hashlib
from collections import defaultdict
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import AppleSupportAgent


def compute_item_hash(
    item_id: str,
    query: str,
    gold_intent: str,
    gold_escalation: str,
    candidate_reply: str,
    candidate_escalation: str,
    rubric_version: str = "v1.2"
) -> str:
    raw_payload = f"{item_id}|||{query.strip()}|||{gold_intent}|||{gold_escalation}|||{candidate_reply.strip()}|||{candidate_escalation}|||{rubric_version}"
    return hashlib.sha256(raw_payload.encode('utf-8')).hexdigest()

def freeze_subset():
    gold_path = Path("data/golden_eval_set.json")
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)

    # Stratified bucket sampling to reach 50 items
    # Group by (gold_intent, gold_escalation, difficulty)
    strata = defaultdict(list)
    for idx, item in enumerate(gold):
        key = (item["gold_intent"], item["gold_escalation"], item.get("difficulty", "MEDIUM"))
        strata[key].append(item)

    selected_items = []
    # Deterministic round-robin pick across sorted strata
    sorted_strata_keys = sorted(strata.keys())
    
    # Target 50 items proportional to strata
    total_needed = 50
    counts = {k: max(1, round(len(v) / len(gold) * total_needed)) for k, v in strata.items()}
    # Adjust to exactly 50
    current_total = sum(counts.values())
    diff = total_needed - current_total
    if diff != 0:
        # adjust largest stratum
        largest_key = max(counts.keys(), key=lambda k: counts[k])
        counts[largest_key] += diff

    for k in sorted_strata_keys:
        needed = counts[k]
        items_in_stratum = strata[k]
        # Pick deterministically with fixed step
        step = max(1, len(items_in_stratum) // needed) if needed > 0 else 1
        picked = items_in_stratum[::step][:needed]
        selected_items.extend(picked)

    # Ensure exactly 50
    if len(selected_items) > 50:
        selected_items = selected_items[:50]
    elif len(selected_items) < 50:
        remaining = [x for x in gold if x not in selected_items]
        selected_items.extend(remaining[:50 - len(selected_items)])

    # Sort by ID for deterministic order
    selected_items = sorted(selected_items, key=lambda x: int(x["id"].replace("GOLD_", "")))

    print(f">> Initializing Proposed AI Agent to generate candidate responses for N={len(selected_items)} subset...")
    agent = AppleSupportAgent().initialize()

    frozen_records = []
    for item in selected_items:
        msg = item["current_customer_message"]
        out = agent.process_message(msg)
        
        c_reply = out["draft_reply"]
        c_esc = out["escalation_decision"]
        c_reason = out["escalation_reason"]
        
        inp_hash = compute_item_hash(
            item_id=item["id"],
            query=msg,
            gold_intent=item["gold_intent"],
            gold_escalation=item["gold_escalation"],
            candidate_reply=c_reply,
            candidate_escalation=c_esc,
            rubric_version="v1.2"
        )
        
        frozen_records.append({
            "item_id": item["id"],
            "conversation_id": item.get("conversation_id", ""),
            "customer_query": msg,
            "gold_intent": item["gold_intent"],
            "gold_escalation": item["gold_escalation"],
            "gold_escalation_reason": item.get("gold_escalation_reason", ""),
            "reference_resolution": item.get("reference_resolution", ""),
            "difficulty": item.get("difficulty", "MEDIUM"),
            "candidate_reply": c_reply,
            "candidate_escalation": c_esc,
            "candidate_reason": c_reason,
            "input_hash": inp_hash,
            "grounded_in": out.get("grounded_in", []),
            "used_evidence": out.get("used_evidence", []),
            "rubric_version": "v1.2"
        })

    out_path = Path("data/frozen_eval_subset_n50.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(frozen_records, f, indent=2, ensure_ascii=False)

    print(f"[OK] Successfully froze {len(frozen_records)} stratified records with SHA256 hashes to {out_path}.")

if __name__ == "__main__":
    freeze_subset()
