import json

with open('data/golden_eval_set.json', 'r', encoding='utf-8') as f:
    gold = json.load(f)

# 1. Update GOLD_194 through GOLD_200
updates_194_200 = {
    "GOLD_194": {
        "conversation_id": "f0d8627aaa32db1d5af320bbed5d3e51",
        "current_customer_message": "#ios11 drains the battery really fast!!😰😭 had to charge 2 x a day compared to just 1 before the update @115858 same w/ my husbands ip7 😭",
        "context_history": [
            {"role": "support", "text": "Let's help improve this. Check out the suggestions shown here: https://t.co/bivpdfBNJ6 but keep us posted."}
        ],
        "customer_query": "#ios11 drains the battery really fast!!😰😭 had to charge 2 x a day compared to just 1 before the update @115858 same w/ my husbands ip7 😭",
        "gold_intent": "BATTERY_AND_HARDWARE",
        "gold_escalation": "AUTO_HANDLE",
        "gold_escalation_reason": "Standard power optimization and battery diagnostic guide can resolve rapid drain.",
        "reference_resolution": "Provide official battery optimization link (https://support.apple.com/HT208387) and troubleshooting steps.",
        "difficulty": "EASY"
    },
    "GOLD_195": {
        "conversation_id": "c9affddbe3e7a3925cc16c9b515962f0",
        "current_customer_message": "plz can u help as the iOS 11.0.2 is bad also ... some apps blocked automatically",
        "context_history": [
            {"role": "customer", "text": "Messenger , FB , photos & messages"},
            {"role": "customer", "text": "Freezing and no response so it must b restarted to go on again"},
            {"role": "customer", "text": "After the latest update 11.0.3 the iPhone still freezing!!!!! I wish u back iOS 10.0.3 again and BTW it's 7+ bought 6 months ago"}
        ],
        "customer_query": "plz can u help as the iOS 11.0.2 is bad also ... some apps blocked automatically Messenger , FB , photos & messages Freezing and no response so it must b restarted to go on again After the latest update 11.0.3 the iPhone still freezing!!!!! I wish u back iOS 10.0.3 again and BTW it's 7+ bought 6 months ago",
        "gold_intent": "IOS_SOFTWARE_UPDATE",
        "gold_escalation": "ESCALATE",
        "gold_escalation_reason": "Persistent app freezing and repeated device restarts across iOS 11.0.2 and 11.0.3 require senior specialist escalation.",
        "reference_resolution": "Acknowledge repeated unresolved freezing and route to senior tier-2 iOS specialist via DM.",
        "difficulty": "HARD"
    },
    "GOLD_196": {
        "conversation_id": "a344ec87129977ba4e55c43525c75f29",
        "current_customer_message": "iOS 11 sucks! Always having to reboot now; apps are constantly crashing. Fix or I'll switch to android.",
        "context_history": [],
        "customer_query": "iOS 11 sucks! Always having to reboot now; apps are constantly crashing. Fix or I'll switch to android.",
        "gold_intent": "CUSTOMER_FEEDBACK_COMPLAINT",
        "gold_escalation": "ESCALATE",
        "gold_escalation_reason": "Customer threatening churn and expressing severe dissatisfaction over crashing apps requires senior empathetic escalation.",
        "reference_resolution": "Empathetic de-escalation addressing churn risk and inviting to DM for personalized diagnosis.",
        "difficulty": "HARD"
    },
    "GOLD_197": {
        "conversation_id": "3b11aa4792502dae5c0d85168cf98a04",
        "current_customer_message": "Upgraded to iOS 11.0.2 & Apple Music lost everything I've added to my library... also won't let me add anything. How do I fix?",
        "context_history": [],
        "customer_query": "Upgraded to iOS 11.0.2 & Apple Music lost everything I've added to my library... also won't let me add anything. How do I fix?",
        "gold_intent": "APP_STORE_AND_BILLING",
        "gold_escalation": "AUTO_HANDLE",
        "gold_escalation_reason": "iCloud Music Library sync toggle and Apple Music account status check can resolve missing library items.",
        "reference_resolution": "Advise checking Settings > Music > iCloud Music Library toggle and confirming subscription status.",
        "difficulty": "MEDIUM"
    },
    "GOLD_198": {
        "conversation_id": "6b33856bb02261e07cdd1265304759c1",
        "current_customer_message": "fuck u my phone just went from 20% to 5% u done fucked everyone up w ios 11",
        "context_history": [],
        "customer_query": "fuck u my phone just went from 20% to 5% u done fucked everyone up w ios 11",
        "gold_intent": "CUSTOMER_FEEDBACK_COMPLAINT",
        "gold_escalation": "ESCALATE",
        "gold_escalation_reason": "Abusive language and severe customer hostility require immediate human de-escalation.",
        "reference_resolution": "De-escalate professionally without arguing and route to specialist via secure DM.",
        "difficulty": "HARD"
    },
    "GOLD_199": {
        "conversation_id": "971c7328ada1a33481b6066bd37e5bfe",
        "current_customer_message": "upgrade to iOS 11 caused a brand new IPhone to draw all battery in 5 hrs without using the phone! WTF",
        "context_history": [],
        "customer_query": "upgrade to iOS 11 caused a brand new IPhone to draw all battery in 5 hrs without using the phone! WTF",
        "gold_intent": "BATTERY_AND_HARDWARE",
        "gold_escalation": "AUTO_HANDLE",
        "gold_escalation_reason": "Post-update background indexing and battery diagnostic steps address new device battery drain.",
        "reference_resolution": "Explain 48-hour post-update re-indexing and provide battery diagnostic steps.",
        "difficulty": "MEDIUM"
    },
    "GOLD_200": {
        "conversation_id": "cfe4a343f0260c4465f7e49eccb46bf8",
        "current_customer_message": "get rid of this for me I don't want it!!!!! https://t.co/c9eA2JwCmu",
        "context_history": [],
        "customer_query": "get rid of this for me I don't want it!!!!! https://t.co/c9eA2JwCmu",
        "gold_intent": "DEVICE_SETUP_AND_USAGE",
        "gold_escalation": "AUTO_HANDLE",
        "gold_escalation_reason": "Simple iPad Dock customization and app removal instructions resolve the user request.",
        "reference_resolution": "Provide instructions for dragging apps off the iPad Dock in Settings > Multitasking & Dock.",
        "difficulty": "EASY"
    }
}

# 2. Fix the 6 battery items previously tagged as APP_STORE_AND_BILLING
battery_fixes = {
    "GOLD_023": {
        "gold_intent": "CUSTOMER_FEEDBACK_COMPLAINT",
        "gold_escalation": "ESCALATE",
        "gold_escalation_reason": "Customer expresses severe ongoing dissatisfaction with battery life and unhelpful previous attempts."
    },
    "GOLD_049": {
        "gold_intent": "BATTERY_AND_HARDWARE",
        "gold_escalation": "ESCALATE",
        "gold_escalation_reason": "Profanity and abnormal hardware battery discharge while plugged into charger."
    },
    "GOLD_055": {
        "gold_intent": "CUSTOMER_FEEDBACK_COMPLAINT",
        "gold_escalation": "ESCALATE",
        "gold_escalation_reason": "Customer details product abandonment (sold iPhone 7 Plus for Galaxy S8) and critical workplace iPad battery failure."
    },
    "GOLD_083": {
        "gold_intent": "BATTERY_AND_HARDWARE",
        "gold_escalation": "AUTO_HANDLE",
        "gold_escalation_reason": "Hardware charging diagnostic steps (checking cable, wall adapter, and charging port debris) apply."
    },
    "GOLD_119": {
        "gold_intent": "BATTERY_AND_HARDWARE",
        "gold_escalation": "AUTO_HANDLE",
        "gold_escalation_reason": "Standard battery usage diagnostic and power-saving troubleshooting steps apply."
    },
    "GOLD_160": {
        "gold_intent": "IOS_SOFTWARE_UPDATE",
        "gold_escalation": "AUTO_HANDLE",
        "gold_escalation_reason": "Clarifying whether upcoming iOS update resolves battery drain and advising current optimization steps."
    }
}

for item in gold:
    gid = item['id']
    if gid in updates_194_200:
        item.update(updates_194_200[gid])
    elif gid in battery_fixes:
        item.update(battery_fixes[gid])

with open('data/golden_eval_set.json', 'w', encoding='utf-8') as f:
    json.dump(gold, f, indent=2, ensure_ascii=False)

print("Updated data/golden_eval_set.json successfully.")
