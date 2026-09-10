import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('evaluation/error_analysis.json', 'r', encoding='utf-8') as f:
    errors = json.load(f)

print(f"Total genuine error records: {len(errors)}")

# Group by failure category:
# 1. Intent confusion: Customer complaint vs software update
# 2. Escalation false positive (over-escalation on repeated attempt/clarification)
# 3. Escalation false negative (under-escalation on subtle distress or multi-issue)
# 4. Multilingual localization
# 5. Device/hardware vs software boundary

selected = []
for err in errors:
    gid = err['item_id']
    # Category 1: GOLD_002 - LTE Watch complaint classified as software update
    if gid == 'GOLD_002':
        selected.append(("Ambiguous Intent (Frustrated Customer vs OS Update)", err))
    # Category 2: GOLD_004 - Spanish multilingual query missed escalation
    elif gid == 'GOLD_004':
        selected.append(("Multilingual Out-of-Scope Routing Miss", err))
    # Category 3: GOLD_008 - Over-escalation due to repeated attempt phrasing
    elif gid == 'GOLD_008':
        selected.append(("False-Positive Escalation (Aggressive Repeat Policy)", err))
    # Category 4: GOLD_019 - macOS display freeze classified as generic usage
    elif gid == 'GOLD_019':
        selected.append(("Cross-Platform System Hang Misclassification", err))
    # Category 5: GOLD_195 - iOS freezing cascade requiring tier-2 escalation
    elif gid == 'GOLD_195':
        selected.append(("Multi-Turn Compounding Failure Cascading", err))

for cat, err in selected:
    print(f"\n=== [{err['item_id']}] {cat} ===")
    print(f"Query: {err['customer_query']}")
    print(f"Gold Intent: {err['gold_intent']} | Pred Intent: {err['predicted_intent']} (Conf: {err['intent_confidence']:.2f})")
    print(f"Gold Escalation: {err['gold_escalation']} | Pred Escalation: {err['predicted_escalation']}")
    print(f"Policy Triggered: {err['policy_triggered']}")
    print(f"Agent Escalation Reason: {err['agent_escalation_reason']}")
    print(f"Draft Reply: {err['draft_reply']}")
