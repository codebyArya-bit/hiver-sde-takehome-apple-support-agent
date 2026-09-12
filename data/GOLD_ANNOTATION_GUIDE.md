# Golden Evaluation Set Annotation Guide & Specification

This document defines the formal annotation standards, intent taxonomy boundaries, escalation criteria, and quality assurance guidelines used to produce `data/golden_eval_set.json` and `data/gold_annotations.csv`.

---

## 1. Objective and Provenance

The Golden Evaluation Set consists of **200 hand-verified, stratified interactions** targeting `@AppleSupport` on Twitter, sourced from the public Customer Support on Twitter dataset (`thoughtvector/customer-support-on-twitter`).

* **Candidate Author Annotation**: All annotations, intent taxonomy assignments, and escalation decisions were performed directly by the **candidate/author** (not an external domain panel or synthetic LLM).
* **Frozen Benchmark**: Gold labels and reference resolutions were **frozen before final benchmark evaluation** to ensure unbiased model comparison across the two baselines and the proposed agent.
* **Single-Turn Evaluation vs. Audit Context**: The evaluated text represents exclusively the incoming single-turn customer message (`current_customer_message` / `customer_query`). Any subsequent conversation turns (`full_thread_for_audit`) are stored strictly for post-hoc audit/reference and are **never provided to the model** during benchmark execution.
* **Zero Synthetic Contamination**: All 200 records represent authentic customer inquiries. Synthetic edge cases (adversarial prompts, prompt injection, simulated extreme threats) are strictly segregated into `data/adversarial_stress_test.json`.
* **Thread-Level Disjointness**: The 200-item gold holdout is thread-disjoint from both the 600 training examples (`apple_train_set.json`) and the 1,000-item historical retrieval KB (`apple_support_kb.json`). The 600 classifier-training conversations are included within the 1,000-item historical KB.

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
| `OUT_OF_SCOPE_OTHER` | Multilingual inquiries, non-Apple products, unintelligible spam | Pure non-English messages (Spanish, French), inquiries about Samsung/Windows, random gibberish | Pure non-English queries requiring translation must be flagged for language localization escalation (`OUT_OF_SCOPE_OTHER` + `ESCALATE`). However, mixed-language messages that already contain a complete, unambiguous English translation (e.g., `GOLD_147` with Russian preamble + full English text) may be classified and handled directly from their English content without requiring localization escalation. |

---

## 3. Escalation Decision Guidelines

Triage decisions must balance safety, legal compliance, and customer trust against operational agent capacity:

### Must `ESCALATE`:
1. **Security & Identity**: Account lockout, credential resets, lost two-factor tokens. (Public social channels must never handle PII or authentication).
2. **Financial Transactions**: Refund requests, duplicate card billings, subscription disputes. (Requires authenticated transaction logs).
3. **Physical & Safety Hazards**: Swollen batteries, overheating causing smoke/burns, shattered glass risks.
4. **Severe Dissatisfaction & Legal/PR Risk**: Threatening litigation, multi-week unresolved delays with prior AppleCare case IDs, abusive rants.
5. **Language Localization**: Inquiries written purely in languages other than English requiring native-language support specialists. *(Note: Mixed-language inquiries containing a full, intelligible English translation such as `GOLD_147` may be auto-handled in English if the inquiry is routine).*
6. **Low Model Confidence**: Inquiries with ambiguous symptoms where automated guidance risks incorrect advice.

### Must `AUTO_HANDLE`:
1. **Deterministic How-To Guidance**: UI navigation, settings toggles, feature discovery with official user guide links.
2. **Standard Software Troubleshooting**: Force restart, storage inspection, update installation guides (`support.apple.com/HT204204`).
3. **Standard Battery Diagnostics**: Advising battery health checks (*Settings > Battery > Battery Health*) and background refresh management.

---

## 4. Canonical Annotation Schema

The exact schema is synchronized across both `data/gold_annotations.csv` and `data/golden_eval_set.json`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `item_id` / `id` | string | Unique identifier (`GOLD_001` through `GOLD_200`). |
| `conversation_id` | string | Unique Twitter conversation thread hash. |
| `customer_query` | string | Incoming single-turn customer inquiry text (identically mapped as `current_customer_message`). |
| `current_customer_message` | string | Canonical incoming customer turn, ensuring strict single-turn evaluation. |
| `gold_intent` | string | Ground truth intent from the 7-class taxonomy. |
| `gold_escalation` | string | Ground truth triage decision (`AUTO_HANDLE` or `ESCALATE`). |
| `gold_escalation_reason` | string | Explicit business justification for the triage decision. |
| `reference_resolution` | string | Target troubleshooting procedure or official canonical URL. |
| `difficulty` | string | Complexity tier: `EASY` (27.0%), `MEDIUM` (55.5%), `HARD` (17.5%). |
| `verification_status` | string | `MANUALLY_AUDITED` across all 200 items. |
| `annotator` | string | `candidate_author` (Candidate / Author audit; not an external expert). |
| `full_thread_for_audit` | list | Multi-turn dialogue history stored exclusively for post-hoc audit/reference (JSON only). |

