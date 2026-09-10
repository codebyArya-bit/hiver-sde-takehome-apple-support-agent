import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/golden_eval_set.json', 'r', encoding='utf-8') as f:
    gold = json.load(f)

for idx, item in enumerate(gold):
    q = item['customer_query'].lower()
    intent = item['gold_intent']
    esc = item['gold_escalation']
    
    # Check if there is an obvious contradiction:
    issues = []
    if intent == 'APP_STORE_AND_BILLING':
        if not any(w in q for w in ['bill', 'subscri', 'pay', 'card', 'refund', 'purchase', 'bought', 'charge', 'money', 'cost', 'store', 'itunes', 'music account']):
            issues.append("APP_STORE_AND_BILLING with no financial/store keywords")
        if any(w in q for w in ['battery', 'charger', 'batt ', 'drain']) and not any(w in q for w in ['refund', 'money', 'subscri']):
            issues.append("Contains battery keywords but labeled APP_STORE_AND_BILLING")
            
    if intent == 'APPLE_ID_AND_ICLOUD':
        if not any(w in q for w in ['icloud', 'apple id', 'appleid', 'password', 'login', 'locked', 'account', 'sign in', 'email']):
            issues.append("APPLE_ID_AND_ICLOUD with no auth/account keywords")

    if intent == 'BATTERY_AND_HARDWARE':
        if not any(w in q for w in ['battery', 'charge', 'charging', 'power', 'screen', 'crack', 'hardware', 'cable', 'port', 'hot', 'heat', 'warm', 'physic', 'drain', 'die', 'dead', 'shut off']):
            issues.append("BATTERY_AND_HARDWARE with no battery/hardware keywords")

    if issues:
        print(f"[{item['id']}] Intent: {intent} | Issues: {issues}")
        print(f"  Query: {item['customer_query'][:100]}\n")
