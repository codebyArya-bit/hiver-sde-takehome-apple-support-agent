# Golden Evaluation Set Annotation Guide & Specification

This document defines the formal annotation standards, intent taxonomy boundaries, escalation criteria, and quality assurance guidelines used to produce `data/golden_eval_set.json` and `data/gold_annotations.csv`.

---

## 1. Objective and Provenance

The Golden Evaluation Set consists of **200 hand-verified, stratified interactions** targeting `@AppleSupport` on Twitter, sourced from the public Customer Support on Twitter dataset (`thoughtvector/customer-support-on-twitter`).

* **Zero Synthetic Contamination**: All 200 records represent authentic customer inquiries. Synthetic edge cases (adversarial prompts, prompt injection, simulated extreme threats) are strictly segregated into `data/adversarial_stress_test.json` so they do not artificially distort benchmark metrics.
* **Thread-Level Disjointness**: Every record belongs to a distinct conversation thread with **zero thread overlap** against the 600-thread training split (`apple_train_set.json`) and the 1,000-thread historical knowledge base (`apple_support_kb.json`).
* **Turn Integrity**: The evaluated text represents the incoming customer turn (`current_customer_message`). Evaluator agents are strictly barred from peeking at future customer turns or subsequent support follow-ups.

---

## 2. Intent Taxonomy & Annotation Boundaries

Each message is assigned to one of seven mutually exclusive operational domains:

| Intent Class | Description | Canonical Symptoms & Inquiries | Exclusions / Boundary Disambiguation |
| :--- | :--- | :--- | :--- |
| `IOS_SOFTWARE_UPDATE` | OS updates, system freezes, installation hangs, release-related bugs | iOS 11 update freezing, boot loop, software update failure, lag after update | If customer complains of battery drain *solely* as battery performance without citing OS release, label `BATTERY_AND_HARDWARE`. |
| `BATTERY_AND_HARDWARE` | Power drain, charging failure, swollen battery, physical screen crack, audio failure | Battery dying rapidly, phone won't charge, screen flickering, cracked glass, hardware heat | If physical heat includes smoke, smell, or swelling, escalate immediately under safety policies. |
| `APPLE_ID_AND_ICLOUD` | Identity, authentication, 2FA, password recovery, iCloud storage syncing | Locked Apple ID, forgotten password, 2FA code not received, iCloud photo sync | Financial billing questions related to iCloud subscription tiers belong to `APP_STORE_AND_BILLING`. |
| `APP_STORE_AND_BILLING` | Purchases, refund requests, disputed subscriptions, credit card charges | Unwanted iTunes charge, refund for in-app purchase, student discount pricing | Technical App Store download freezing belongs to `IOS_SOFTWARE_UPDATE`. |
| `DEVICE_SETUP_AND_USAGE` | How-to guidance, UI navigation, feature discovery, settings toggles | Orientation lock toggle, iPad dock removal, Apple Watch pairing, AirDrop setup | If user asks how to recover an account, route to `APPLE_ID_AND_ICLOUD`. |
| `CUSTOMER_FEEDBACK_COMPLAINT` | Expressed dissatisfaction, service delays, store experience, brand frustration | Waiting 2 weeks for AppleCare callback, sarcastic complaints, threats to switch to Android | Routine technical questions with mild annoyance still classify by technical intent unless primary intent is venting/complaint. |
| `OUT_OF_SCOPE_OTHER` | Multilingual inquiries, non-Apple products, unintelligible spam | Non-English messages (Spanish, French), inquiries about Samsung/Windows, random gibberish | Non-English queries must be flagged for language localization escalation. |

---

## 3. Escalation Decision Guidelines

Triage decisions must balance safety, legal compliance, and customer trust against operational agent capacity:

### Must `ESCALATE`:
1. **Security & Identity**: Account lockout, credential resets, lost two-factor tokens. (Public social channels must never handle PII or authentication).
2. **Financial Transactions**: Refund requests, duplicate card billings, subscription disputes. (Requires authenticated transaction logs).
3. **Physical & Safety Hazards**: Swollen batteries, overheating causing smoke/burns, shattered glass risks.
4. **Severe Dissatisfaction & Legal/PR Risk**: Threatening litigation, multi-week unresolved delays with prior AppleCare case IDs, abusive rants.
5. **Language Localization**: Inquiries written in languages other than English requiring native-language support specialists.
6. **Low Model Confidence**: Inquiries with ambiguous symptoms where automated guidance risks incorrect advice.

### Must `AUTO_HANDLE`:
1. **Deterministic How-To Guidance**: UI navigation, settings toggles, feature discovery with official user guide links.
2. **Standard Software Troubleshooting**: Force restart, storage inspection, update installation guides (`support.apple.com/HT204204`).
3. **Standard Battery Diagnostics**: Advising battery health checks (*Settings > Battery > Battery Health*) and background refresh management.

---

## 4. Annotation Schema (`data/gold_annotations.csv`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | string | Unique identifier (`GOLD_001` through `GOLD_200`). |
| `conversation_id` | string | Unique Twitter conversation thread hash. |
| `current_customer_message` | string | Exact incoming customer inquiry text. |
| `gold_intent` | string | Ground truth intent from the 7-class taxonomy. |
| `gold_escalation` | string | Ground truth triage decision (`AUTO_HANDLE` or `ESCALATE`). |
| `gold_escalation_reason` | string | Explicit business justification for the triage decision. |
| `reference_resolution` | string | Target troubleshooting procedure or official canonical URL. |
| `difficulty` | string | Complexity tier: `EASY` (27.0%), `MEDIUM` (55.5%), `HARD` (17.5%). |
| `annotator` | string | Verified auditor identifier. |
| `annotation_date` | string | Audit timestamp. |
| `verification_status` | string | `MANUALLY_CONFIRMED` across all 200 items. |
