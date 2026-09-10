import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/golden_eval_set.json', 'r', encoding='utf-8') as f:
    gold = json.load(f)

# Group by intent
by_intent = {}
for item in gold:
    by_intent.setdefault(item['gold_intent'], []).append(item)

for intent, items in by_intent.items():
    print(f"\n==================== {intent} ({len(items)}) ====================")
    for it in items:
        esc = it['gold_escalation']
        q = it['customer_query'].replace('\n', ' ')
        if len(q) > 85:
            q = q[:85] + "..."
        print(f"[{it['id']}] [{esc}] {q}")
