# Sampling and Labeling Methodology: Golden Evaluation Set

## 1. Overview
The Golden Evaluation Set comprises **200 hand-curated and annotated customer support interactions** targeting `@AppleSupport` on Twitter. This benchmark serves as the authoritative ground truth for assessing intent classification, grounded reply drafting, and escalation triage.

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
3. **Training Set**: 600 complete conversation threads partitioned into `data/processed/apple_train_set.json` (derived from the training slice).
4. **Golden Evaluation Set**: Exactly 200 complete, distinct conversation threads held out into `data/golden_eval_set.json`.
5. **Verified Overlap**: Thread ID overlap between training and golden evaluation sets is **strictly 0**.

---

## 3. Stratification & Diversity Dimensions (N = 200)

1. **Intent Distribution**:
   - `IOS_SOFTWARE_UPDATE` (39.0%, 78 items): OS freezes, update installation loops, Bluetooth/Wi-Fi drops.
   - `BATTERY_AND_HARDWARE` (24.0%, 48 items): Rapid battery drain, physical damage, cracked screen, swollen battery.
   - `DEVICE_SETUP_AND_USAGE` (14.5%, 29 items): UI navigation (rotation lock), data transfer, control center setup.
   - `CUSTOMER_FEEDBACK_COMPLAINT` (8.5%, 17 items): Dissatisfaction with service, store wait times, rants.
   - `APPLE_ID_AND_ICLOUD` (6.5%, 13 items): Locked Apple IDs, lost 2FA codes, authentication loops.
   - `APP_STORE_AND_BILLING` (5.5%, 11 items): In-app purchase refunds, unauthorized subscription charges.
   - `OUT_OF_SCOPE_OTHER` (2.0%, 4 items): Off-topic queries, multi-lingual messages (Spanish/French), and noise.

2. **Escalation Distribution**:
   - **`AUTO_HANDLE` (71.0%, 142 items)**: Resolvable with standard public troubleshooting workflows, settings navigation, or official knowledge base links.
   - **`ESCALATE` (29.0%, 58 items)**: Requiring human intervention due to account authentication, financial transactions, physical hardware inspection, safety hazards, high customer distress, or language barriers.

3. **Difficulty Profiling**:
   - **`EASY` (25.5%, 51 items)**: Explicit, single-intent queries with direct keywords.
   - **`MEDIUM` (59.5%, 119 items)**: Real-world queries with noisy syntax, minor typos, or implicit technical context.
   - **`HARD` / `EDGE_CASE` (15.0%, 30 items)**: Multi-intent compound questions, severe emotional distress, legal threats, safety hazards (swollen batteries), or sarcasm.

---

## 4. Human Benchmark Annotation Protocol

Each item was annotated with:
- `gold_intent`: Target class from the 7-intent taxonomy.
- `gold_escalation`: Binary triage decision (`AUTO_HANDLE` vs `ESCALATE`).
- `gold_escalation_reason`: Explicit business rationale.
- `reference_resolution`: Canonical troubleshooting instructions and target Apple Support links.
- `human_rubric`: Blind human ratings (1.0 to 5.0) across Groundedness, Brand Voice, Actionability, and Escalation Appropriateness, reflecting realistic difficulty variation.
