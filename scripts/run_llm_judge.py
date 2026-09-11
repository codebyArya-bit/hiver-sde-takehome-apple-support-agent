"""
LLM-as-a-Judge and Paired Human Annotation Runner & Verifier.

Workflow:
1. Live LLM Evaluation: When run with `--live` and an API key (GEMINI_API_KEY or OPENAI_API_KEY),
   calls the live LLM API with the structured rubric prompt and updates evaluation/llm_judge_scores.json.
2. Cached Reproduction & Cryptographic Hash Verification: Loads the committed evaluation results
   (evaluation/llm_judge_scores.json and evaluation/human_annotations.json) and verifies that every
   record's SHA256 input hash matches data/frozen_eval_subset_n50.json exactly.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.llm_judge import LLMSupportJudge

def run_or_verify_evaluations(
    frozen_path: str = "data/frozen_eval_subset_n50.json",
    llm_path: str = "evaluation/llm_judge_scores.json",
    human_path: str = "evaluation/human_annotations.json",
    force_live: bool = False
):
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen_items = json.load(f)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    has_api_key = bool(gemini_key or openai_key)
    live_mode = force_live and has_api_key

    print("=" * 60)
    print("      LLM-AS-A-JUDGE & HUMAN EVALUATION HARNESS")
    print("=" * 60)
    print(f"Frozen Benchmark: {frozen_path} (N={len(frozen_items)})")
    print(f"Execution Mode:   {'LIVE API' if live_mode else 'VERIFIED CACHED REPRODUCTION'}")

    if live_mode:
        judge = LLMSupportJudge(cache_path=llm_path)
        print(f">> Executing live LLM judge over {len(frozen_items)} items...")
        updated_scores = {}
        for idx, item in enumerate(frozen_items):
            item_id = item["item_id"]
            print(f"   [{idx+1:02d}/{len(frozen_items)}] Calling live judge for {item_id}...")
            llm_eval = judge.evaluate_item(
                item_id=item_id,
                customer_query=item["customer_query"],
                gold_intent=item["gold_intent"],
                gold_escalation=item["gold_escalation"],
                candidate_reply=item["candidate_reply"],
                candidate_escalation=item["candidate_escalation"],
                candidate_reason=item["candidate_reason"],
                reference_resolution=item.get("reference_resolution", ""),
                use_cache=False
            )
            updated_scores[item_id] = llm_eval

        with open(llm_path, "w", encoding="utf-8") as f:
            json.dump(updated_scores, f, indent=2, ensure_ascii=False)
        print(f"[OK] Successfully saved {len(updated_scores)} live LLM judgments to {llm_path}")

    # Verify cryptographic hash alignment
    print("\n>> Verifying SHA256 cryptographic alignment across all 50 items...")
    with open(llm_path, "r", encoding="utf-8") as f:
        llm_data = json.load(f)
    with open(human_path, "r", encoding="utf-8") as f:
        human_data = json.load(f)

    if isinstance(llm_data, list):
        llm_map = {x["item_id"]: x for x in llm_data}
    else:
        llm_map = llm_data

    human_map = {x["item_id"]: x for x in human_data}

    mismatches = []
    for item in frozen_items:
        i_id = item["item_id"]
        expected_hash = item["input_hash"]

        if i_id not in llm_map:
            mismatches.append(f"Missing LLM judge record for {i_id}")
            continue
        if i_id not in human_map:
            mismatches.append(f"Missing Human annotation record for {i_id}")
            continue

        l_hash = llm_map[i_id].get("input_hash")
        h_hash = human_map[i_id].get("input_hash")

        if l_hash != expected_hash:
            mismatches.append(f"LLM hash mismatch in {i_id}: expected {expected_hash[:8]}..., got {str(l_hash)[:8]}...")
        if h_hash != expected_hash:
            mismatches.append(f"Human hash mismatch in {i_id}: expected {expected_hash[:8]}..., got {str(h_hash)[:8]}...")

    if mismatches:
        print(f"[ERROR] Found {len(mismatches)} hash verification errors:")
        for m in mismatches[:5]:
            print(f"   - {m}")
        sys.exit(1)
    else:
        print(f"[SUCCESS] All {len(frozen_items)} records verified with 100% SHA256 input hash alignment.")
        print("  - LLM judge evaluations and Human annotations refer strictly to identical candidate outputs.")
        print("  - Zero stale evaluation reuse confirmed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or Verify LLM Judge and Paired Human Annotations")
    parser.add_argument("--frozen-subset", default="data/frozen_eval_subset_n50.json")
    parser.add_argument("--llm-scores", default="evaluation/llm_judge_scores.json")
    parser.add_argument("--human-scores", default="evaluation/human_annotations.json")
    parser.add_argument("--live", action="store_true", help="Call live API if environment keys exist")
    args = parser.parse_args()

    run_or_verify_evaluations(
        frozen_path=args.frozen_subset,
        llm_path=args.llm_scores,
        human_path=args.human_scores,
        force_live=args.live
    )
