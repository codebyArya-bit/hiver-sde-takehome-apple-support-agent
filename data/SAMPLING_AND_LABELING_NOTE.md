# Sampling and Labeling Methodology: Golden Evaluation Set

## 1. Overview
The Golden Evaluation Set comprises **200 hand-curated, stratified, and verified customer support interactions** targeting `@AppleSupport` on Twitter, sourced from the Kaggle Customer Support on Twitter dataset. Every record is 100% genuine real-world Twitter data. This benchmark serves as the authoritative ground truth for assessing intent classification, grounded reply drafting, and auto-handle vs. escalation triage.

Adversarial synthetic cases (e.g. prompt injection, jailbreak attempts, edge synthetic queries) are strictly separated into `data/adversarial_stress_test.json` so they do not contaminate the primary real-data benchmark.

---

## 2. Zero-Leakage Thread-Disjoint Sampling Strategy

### The Risk of Conversational Contamination
A critical pitfall in conversational NLP evaluation is **intra-thread data leakage**:
* Suppose a customer posts Tweet A, the brand replies with Tweet B, and the customer follows up with Tweet C.
* If Tweet A is included in the training set or RAG vector index, and Tweet C lands in the evaluation set, the evaluation is severely contaminated. The model can simply memorize customer-specific entity names, device details, and previous troubleshooting outcomes.

### Strict Conversation-Thread Partitioning
To guarantee complete independence and zero data leakage:
1. All tweets were grouped by their unique `conversation_id`.
2. **Knowledge Base (RAG)**: 1,000 complete conversation threads indexed into `data/processed/apple_support_kb.json`.
3. **Training Set**: 600 complete conversation threads partitioned into `data/processed/apple_train_set.json`.
4. **Golden Evaluation Set**: Exactly 200 complete, distinct conversation threads held out into `data/golden_eval_set.json`.
5. **Verified Overlap**: Thread ID overlap between Train, KB, and Golden evaluation sets is **strictly 0** (verified by `scripts/prepare_dataset.py`).

### Turn Separation Architecture
To reflect real-world operational conditions where an agent must classify each incoming message in real time without access to future customer follow-ups:
* `current_customer_message`: The specific incoming customer tweet to be handled.
* `context_history`: Prior dialogue turns in the thread (if any).
* `customer_query`: Synthesized query representation used for evaluation.

---

## 3. Stratification & Diversity Dimensions (N = 200)

1. **Intent Distribution**:
   - `IOS_SOFTWARE_UPDATE` (35.5%, 71 items): OS freezing, update installation loops, cellular/Wi-Fi drops.
   - `BATTERY_AND_HARDWARE` (27.5%, 55 items): Rapid battery drain, physical damage, cracked screen, swollen battery.
   - `CUSTOMER_FEEDBACK_COMPLAINT` (16.0%, 32 items): Dissatisfaction with service, store wait times, rants.
   - `DEVICE_SETUP_AND_USAGE` (11.5%, 23 items): UI navigation (rotation lock, dock), data transfer, control center setup.
   - `APPLE_ID_AND_ICLOUD` (6.0%, 12 items): Locked Apple IDs, lost 2FA codes, authentication loops.
   - `APP_STORE_AND_BILLING` (2.0%, 4 items): In-app purchase refunds, unauthorized subscription charges.
   - `OUT_OF_SCOPE_OTHER` (1.5%, 3 items): Off-topic queries, multi-lingual messages (Spanish/French), and noise.

2. **Escalation Distribution**:
   - **`AUTO_HANDLE` (71.0%, 142 items)**: Resolvable with standard public troubleshooting workflows, settings navigation, or official knowledge base links.
   - **`ESCALATE` (29.0%, 58 items)**: Requiring human intervention due to account authentication, financial transactions, physical hardware inspection, safety hazards, high customer distress, or language barriers.

3. **Difficulty Profiling**:
   - **`EASY` (27.0%, 54 items)**: Explicit, single-intent queries with direct keywords.
   - **`MEDIUM` (55.5%, 111 items)**: Real-world queries with noisy syntax, minor typos, or implicit technical context.
   - **`HARD` / `EDGE_CASE` (17.5%, 35 items)**: Multi-intent compound questions, severe emotional distress, churn threats, safety hazards (swollen batteries), or sarcasm.

---

## 4. Human Benchmark Annotation Protocol

Each item in `data/golden_eval_set.json` was manually verified and annotated with:
- `gold_intent`: Target class from the 7-intent taxonomy.
- `gold_escalation`: Binary triage decision (`AUTO_HANDLE` vs `ESCALATE`).
- `gold_escalation_reason`: Explicit business rationale for triage decision.
- `reference_resolution`: Canonical troubleshooting instructions and target Apple Support links.
- `difficulty`: Complexity rating (`EASY`, `MEDIUM`, `HARD`).

### Separation of Inter-Rater Reliability Protocol
To ensure genuine inter-rater reliability, rubric scoring is not assigned a priori to the query. Instead:
- Frozen agent responses to 50 golden items were scored by:
  1. The candidate author manually scoring 50 frozen outputs using the same rubric without viewing LLM ratings (`evaluation/human_annotations.json`).
  2. Gemini 2.5 Flash operating under the identical multi-dimensional rubric (`evaluation/llm_judge_scores.json`).
- Agreement is measured on the exact same candidate outputs, achieving **Pearson $r = 0.920$**, **Spearman $\rho = 0.766$**, **MAE = 0.226 points** (100% within 0.5 points), and **Cohen's $\kappa = 1.000$** on binary Escalation Appropriateness (with Cohen's $\kappa$ reported as N/A on uniform categories rather than artificial 1.0 substitutions). 100% of pairs match cryptographic SHA256 input hashes.
