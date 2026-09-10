import sys
import json

def audit():
    with open('data/golden_eval_set.json', 'r', encoding='utf-8') as f:
        gold = json.load(f)

    with open('data/gold_audit_review.txt', 'w', encoding='utf-8') as out:
        for idx, item in enumerate(gold):
            out.write(f"[{idx+1}/200] ID: {item['id']} | Intent: {item['gold_intent']} | Esc: {item['gold_escalation']}\n")
            out.write(f"Reason: {item.get('gold_escalation_reason', '')}\n")
            out.write(f"Query: {item['customer_query']}\n")
            out.write("-" * 70 + "\n")
    print("Exported 200 items to data/gold_audit_review.txt")

if __name__ == '__main__':
    audit()
