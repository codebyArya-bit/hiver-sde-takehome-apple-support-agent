# AI Customer Support Agent for @AppleSupport: Comprehensive Technical Report

**Author**: Hiver SDE Intern Candidate  
**Date**: September 2026  
**Target Brand**: Apple Support (`@AppleSupport` on Twitter/X)  
**Dataset**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`, threaded by `TNE-AI`)  
**Holdout Evaluation Set**: 200 Hand-Audited, Stratified Gold Examples (`data/golden_eval_set.json`)  
**Compiled PDF**: [REPORT.pdf](REPORT.pdf) (Strictly <= 6 Pages, Currently 3 Pages)

---

## Executive Summary

Customer support on social media is high-stakes, real-time, and public. For a brand like Apple, an AI support agent must do more than answer questions: it must maintain customer trust, protect user security and data privacy, comply strictly with character limits, and know with certainty **when not to answer**.

This report documents the design, implementation, and empirical evaluation of an **Evaluation-Focused AI Support Agent Prototype with Calibrated Safety Guardrails** for `@AppleSupport`. We evaluate our system against two baselines (a Trivial Majority-Rule Baseline and a Classical Statistical Machine Learning Baseline) on a 200-sample hand-audited Golden Evaluation Set. The holdout gold set is strictly thread-disjoint from both the 600 training threads and the 1,000 historical KB retrieval corpus (with training threads contained within the KB corpus).

The proposed agent achieves:
- **61.5% Out-of-Sample Intent Accuracy** across a 7-class domain taxonomy (vs. 35.0% Trivial and 54.5% Simple Baseline).
- **Calibrated Confidence**: Expected Calibration Error (ECE) of **0.040** (vs. 0.350 Trivial and 0.362 Simple) and Brier Score of **0.607** (vs. 1.000 Trivial and 0.796 Simple).
- **67.3% Escalation Recall** on safety-critical interactions (catching 35 of 52 escalations, vs. **0.0%** for Trivial Baseline and **3.9%** for Simple Baseline).
- **0.612 Escalation F2 Score** and a **50%+ reduction in the illustrative Weighted Risk-Cost Penalty** (128 vs. 260 Trivial and 250 Simple Baseline).
- **79.5% Official Apple Domain Inclusion Rate** and **76.1% Intent-Link Relevance Rate** (vs. 0.0% for baselines).
- **100.0% Twitter Character Limit Compliance** (<280 chars).
- Mean Heuristic Rubric score of **4.64 / 5.00** (vs. 4.39 for Trivial and 4.24 for Simple Baseline).
- Paired Human vs. LLM-as-a-Judge Study ($N=50$): **Pearson $r = 0.940$**, **Spearman $\rho = 0.785$**, **MAE = 0.228 points** (100.0% within 0.5 points), with cryptographic SHA256 input hash verification across all evaluated pairs.
- The full evaluation suite reproduces completely offline on standard CPU in **~6 seconds**, easily satisfying the <15-minute reproduction requirement with zero external API dependencies.

---

## Section 1: Problem Framing & Scope Boundaries

### 1.1 What "Good" Means for @AppleSupport
On Twitter, `@AppleSupport` handles a high volume of public customer inquiries spanning hardware, software, and cloud services. Through analysis of the historical corpus, "good" support for Apple is characterized by four non-negotiable operational principles:

1. **Safety and Privacy First (Non-Negotiable Escalation)**: Public Twitter is an unauthenticated communication channel. Any issue requiring access to Apple ID credentials, two-factor authentication, credit card numbers, or App Store purchase receipts must **never** be resolved in public. "Good" means immediately routing these inquiries to private authenticated channels (DM or official Apple recovery portals) without exposing the user to phishing or data leakage.
2. **Definitive Troubleshooting Steps (Actionability)**: Customers reach out when their devices disrupt their daily lives. Vague guidance ("have you checked your phone?") causes customer frustration. "Good" means providing concrete, deterministic steps (e.g., *Settings > Battery > Battery Health*, or *Settings > General > iPhone Storage*) accompanied by canonical knowledge base links (`support.apple.com/HT...`).
3. **Apple Brand Persona**: Apple's customer service persona is calm, welcoming, professional, and empathetic ("We're here to help", "Let's work together to get this sorted"). The agent must never sound defensive, condescending, or robotic.
4. **Strict Platform & Format Compliance**: Tweets must strictly obey the 280-character ceiling and ensure that links are never broken or truncated mid-URL.

### 1.2 What We Chose NOT to Build (Explicit Non-Goals)
To preserve safety and maintain high signal-to-noise ratio, we made deliberate decisions about what to exclude from the prototype's scope:

* **No Automated Financial Execution (No Auto-Refunds)**: We deliberately do not build automated refund or subscription cancellation execution. Triggering financial payouts from an unauthenticated social media bot creates severe fraud and abuse vulnerabilities. All billing disputes are routed to human specialists and official authenticated portals (`reportaproblem.apple.com`).
* **No Direct Credential or Password Resets in Chat**: The agent will never ask for or accept passwords, passcodes, or 2FA codes. Account recovery is exclusively directed to `iforgot.apple.com` or private DM escalation queues.
* **No Machine-Translated Technical Troubleshooting**: While the agent recognizes non-English incoming queries (e.g. Spanish, French), we explicitly chose not to auto-translate and reply in English or perform machine-translated technical troubleshooting. Untested translations of technical terms (e.g., "DFU mode", "Recovery Key") can cause catastrophic user data loss. These queries are routed directly to regional native-language human queues.
* **No Speculative Hardware Diagnostics or Cost Promises**: If a user reports physical damage (cracked screen, broken microphone, or a bulging battery), the agent never promises warranty coverage or cost estimates. It directs the user to book a Genius Bar appointment or mail-in repair.

---

## Section 2: Results vs. Two Baselines

### 2.1 Baseline Architectures
1. **Trivial Baseline**:
   - Intent: Predicts majority class (`IOS_SOFTWARE_UPDATE`).
   - Escalation: **Always predict `AUTO_HANDLE`** (majority triage class, reflecting 74% of incoming traffic).
   - Reply: Static generic canned macro reply ("Thanks for reaching out! We'd love to help. Please DM us your device model and iOS version so we can look into this.").
2. **Simple Baseline**:
   - Intent: Multinomial Naive Bayes trained on word count vectors.
   - Escalation: Simple keyword-matching rules (looking for words like "refund", "human", "agent", "sue", "manager").
   - Reply: 1-Nearest Neighbor historical retrieval selecting top retrieved resolution text directly.
3. **Proposed AI Agent**:
   - Intent Classifier: Calibrated Logistic Regression over sublinear TF-IDF character and word n-grams with Platt probability scaling.
   - Retrieval Engine: Sparse TF-IDF retrieval over 1,000 historical `@AppleSupport` resolutions with domain-whitelisted URL grounding.
   - Escalation Engine: Asymmetric safety policy rules + calibrated confidence thresholds with explicit stated reasons and recall-prioritized triage.
   - Reply Generator: Brand-conditioned reply drafting enforcing Twitter <280-char constraints with structured `used_evidence` and `grounded_in` tracking.

### 2.2 Benchmark Results Table (Thread-Disjoint Holdout Split)

The 200-item gold holdout is thread-disjoint from both the 600 training examples and the 1,000-item historical retrieval KB. The 600 classifier-training conversations are included within the 1,000-item historical KB:

| Metric Dimension | Trivial Baseline (Always Auto-Handle) | Simple Baseline (Naive Bayes + 1-NN) | Proposed AI Agent (RAG + Policy) | Real-World Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Classification Accuracy** | 35.0% | 54.5% | **61.5%** | Out-of-sample generalization across 7 domain intents |
| **Intent Macro F1** | 0.074 | 0.282 | **0.475** | Balanced performance across minority and majority intents |
| **Brier Calibration Score (Lower=Better)** | 1.000 | 0.796 | **0.607** | Superior statistical probability calibration directly from model |
| **Expected Calibration Error (ECE)** | 0.350 | 0.362 | **0.040** | Highly calibrated confidence (predicted confidence tracks empirical accuracy) |
| **Escalation Decision Accuracy** | 74.0% | 75.0% | **70.0%** | The proposed policy sacrifices raw escalation accuracy and precision to substantially increase recall for cases requiring human intervention |
| **Escalation Recall (Safety-Critical)** | **0.0%** | **3.9%** | **67.3%** | Catches 35 of 52 escalations; baselines miss 96% to 100% |
| **Escalation Precision** | 0.0% | 100.0% | **44.9%** | The safety-oriented operating point catches substantially more required escalations at the cost of additional unnecessary human handoffs (44.9% precision) |
| **Escalation F2 Score (Recall-Weighted)** | 0.000 | 0.048 | **0.612** | Recall weighted 2x vs. precision, reflecting safety-first priority |
| **False Escalation Rate (Lower=Better)**| 0.0% | 0.0% | **29.0%** | Trade-off: accepts ~29% false alarms to protect customer accounts |
| **Illustrative 5:1 Risk Penalty (5*FN + 1*FP)** | 260 | 250 | **128** | Candidate-selected offline penalty reduced by over 48% |
| **SacreBLEU Score** | 0.2 | 0.9 | **4.2** | Lexical overlap with authentic support resolutions |
| **Twitter Char Limit Compliance (<280)** | 100.0% | 95.0% | **100.0%** | Zero tweet truncation or broken URL artifacts |
| **Official Apple Domain Inclusion Rate** | 0.0% | 0.0% | **79.5%** | Verified canonical Apple domains (`support.apple.com`, `iforgot.apple.com`) |
| **Intent-Link Relevance Rate** | 0.0% | 0.0% | **76.1%** | Canonical URL matches classified customer issue domain |
| **Heuristic: Groundedness (1–5)** | 4.20 | 4.06 | **4.78** | Factual grounding in verified historical resolution precedents |
| **Heuristic: Brand Voice & Empathy (1–5)** | 5.00 | 4.36 | **4.58** | Professional, empathetic Apple tone within single-tweet limits |
| **Heuristic: Actionability (1–5)** | 4.20 | 4.33 | **4.80** | Concrete step-by-step guidance and canonical navigation paths |
| **Heuristic: Escalation Appropriateness (1–5)**| 4.14 | 4.18 | **4.40** | Safe triage decisions aligned with safety and compliance policies |
| **Heuristic: Overall Quality Score (1–5)** | 4.39 | 4.24 | **4.64** | Holistic quality superiority over both baselines |

---

## Section 3: Failure Mode Analysis (Top 5 Real Observed Failures)

A transparent post-mortem of our model's errors is essential to earning organizational trust. All 5 cases are genuine observed errors from `evaluation/error_analysis.json`:

### Failure Mode 1: Sarcasm and Ambiguous Frustration [GOLD_002]
* **Customer Query**:
  > *"It's been nearly two weeks and I've yet to get LTE on my Watch () working. Think I paid a premium for a spec of red paint."*
* **Observed Agent Output**: Predicted Intent = `OUT_OF_SCOPE_OTHER` (Confidence: 0.49). Decision = `AUTO_HANDLE` (via `NONE_AUTO_HANDLED`).
* **Gold Ground Truth**: Intent = `CUSTOMER_FEEDBACK_COMPLAINT`, Escalation = `ESCALATE`.
* **Root Cause & Mitigation**: Sarcastic phrasing ("paid a premium for a spec of red paint") lacked overt profanity or legal threat keywords. Since classifier confidence (0.49) was above the low-confidence cutoff (0.40), it fell through to auto-handle. Mitigation: Add multi-week duration indicators ("nearly two weeks", "days now") and device churn phrases to intent-independent escalation rules.

### Failure Mode 2: Multilingual Language Routing Miss [GOLD_004]
* **Customer Query**:
  > *"Mira que me gusta vuestra actualización pero me va como el culo ahora, cuando queráis lo solucionáis."*
* **Observed Agent Output**: Predicted Intent = `OUT_OF_SCOPE_OTHER` (Confidence: 0.69), Decision = `AUTO_HANDLE`.
* **Gold Ground Truth**: Intent = `OUT_OF_SCOPE_OTHER`, Escalation = `ESCALATE` (Non-English localization routing).
* **Root Cause & Mitigation**: Idiomatic European Spanish ("como el culo") was not in the localization keyword regex, causing the query to be classified as out-of-scope and auto-handled in English. Mitigation: Integrate a pre-classification language identification filter (`fastText` or `langdetect`) at the message ingestion boundary.

### Failure Mode 3: False-Positive Escalation on Repeated Troubleshooting Phrasing [GOLD_008]
* **Customer Query**:
  > *"I've had to do it multiple times when I reset my device and how do I change trusted device from old phone to new phone?"*
* **Observed Agent Output**: Decision = `ESCALATE` (Triggered via `POLICY_REPEATED_UNRESOLVED_FAILURE`).
* **Gold Ground Truth**: Decision = `AUTO_HANDLE` (Routine device pairing how-to).
* **Root Cause & Mitigation**: The phrase "multiple times" triggered the repeated failure heuristic on a standard UI how-to query. Mitigation: Condition repeated-failure rules on negative emotional sentiment tokens rather than raw occurrence words alone.

### Failure Mode 4: Low-Confidence Safety Interception [GOLD_019]
* **Customer Query**:
  > *"Cool new feature in macOS High Sierra, it knows you've been working too hard and freezes the screen, but not the mouse."*
* **Observed Agent Output**: Intent = `IOS_SOFTWARE_UPDATE` (Confidence: 0.37). Escalation = `ESCALATE` (via `POLICY_LOW_MODEL_CONFIDENCE`).
* **Gold Ground Truth**: Intent = `IOS_SOFTWARE_UPDATE`, Decision = `AUTO_HANDLE`.
* **Root Cause & Mitigation**: Sarcastic complaint confused the classifier, resulting in low posterior confidence (0.37). The low-confidence safety catch safely escalated the interaction to prevent hallucinatory troubleshooting guidance.

### Failure Mode 5: Sensitive Domain Precautionary Escalation [GOLD_022]
* **Customer Query**:
  > *"Grrr! Why won't the contacts from my phone back up to iCloud?? #icloud"*
* **Observed Agent Output**: Intent = `APPLE_ID_AND_ICLOUD` (Confidence: 0.77). Escalation = `ESCALATE` (via `POLICY_SENSITIVE_DOMAIN`).
* **Gold Ground Truth**: Intent = `APPLE_ID_AND_ICLOUD`, Escalation = `AUTO_HANDLE`.
* **Root Cause & Mitigation**: Broad sensitive domain rules escalate all Apple ID / iCloud inquiries to safeguard account privacy, causing a conservative false escalation on routine contact syncing. Mitigation: Refine sub-domain segmentation between account recovery/credentials and standard iCloud sync settings.

---

## Section 4: "What is Misleading About My Headline Number?" (Mandatory Section)

A candidate who blindly presents high numbers without understanding their operational reality cannot be trusted in production. Here is our rigorous critique:

### 1. The Deception of Raw Accuracy in Imbalanced Triage
The Trivial Baseline achieves **74.0% accuracy** simply by auto-handling everything. However, this exposes the central failure of raw accuracy on imbalanced distributions:
- Trivial Baseline Escalation Recall: **0.0%** (catches 0 out of 52 human escalations).
- Simple Baseline Escalation Recall: **3.9%** (catches 2 out of 52 human escalations).
- Proposed AI Agent Escalation Recall: **67.3%** (catches 35 out of 52 human escalations).
Accuracy is an actively misleading metric in customer support triage. This is why we formalize triage performance via the **Escalation F2 Score (0.612)**, weighting recall twice as heavily as precision.

### 2. Operational Reality of 67.3% Escalation Recall
An escalation recall of 67.3% means that while the agent successfully intercepts 35 safety-critical inquiries, **17 out of 52 escalations are missed** (e.g., subtle sarcasm in GOLD_002 or foreign-language idioms in GOLD_004). In an enterprise production environment, we would **not** enable unrestricted autonomous auto-replies at this recall level. Instead, deployment must follow a disciplined **tri-state routing policy**:
- **High-Confidence Auto-Handle (>0.85 confidence)**: Restricted strictly to deterministic, non-sensitive routine setup and informational queries.
- **Medium-Confidence Agent-Assist Draft (0.55–0.85 confidence)**: Pre-generate grounded draft responses in the Hiver inbox for one-click human support agent review and approval.
- **Low-Confidence / Sensitive Direct Escalation (<0.55 confidence or safety trigger)**: Immediate human routing with zero automated customer-facing replies.

### 3. Illustrative Asymmetric Cost Metric (5:1 Risk Penalty)
Under an illustrative candidate-selected 5:1 penalty ($5 \times \text{FN} + 1 \times \text{FP}$) that weights missed escalations more heavily than unnecessary escalations, the policy reduces the offline penalty from 260 to 128. However, this formula is an illustrative candidate-selected risk weighting, not an empirical dollar model of Hiver's or Apple's actual operational balance sheet.

### 4. Single-Turn Evaluation vs. Dynamic Multi-Turn Conversations
Our benchmark evaluates incoming customer inquiries as single-turn interactions (`current_customer_message`). In reality, customer support dialogues span multiple turns. Offline single-turn evaluation cannot measure downstream resolution rate or customer abandonment when initial troubleshooting fails.

### 5. Intent Accuracy is Bound by Domain Taxonomy Framing
Our out-of-sample intent accuracy is 61.5% across 7 coarse categories. In an unconstrained open-vocabulary setting, intent accuracy would naturally degrade. The metric measures consistency within our defined taxonomy, not general conversational intelligence.

---

## Section 5: Candidate Human Annotator vs. LLM-as-a-Judge Agreement (N=50 Paired Frozen Outputs)

To evaluate automated rubric reliability, we conducted an inter-rater agreement study comparing Gemini 2.5 Flash rubric scoring and candidate author blind scoring on the exact same 50 frozen agent outputs. The candidate manually scored the 50 frozen outputs using the four-axis rubric without viewing the LLM ratings, then compared the two rating sets:

* **Pearson Correlation ($r$)**: **0.940** on Overall Rubric Score, indicating strong linear tracking of human scoring.
* **Spearman Rank Correlation ($\rho$)**: **0.785**, demonstrating consistent ordinal ranking of response quality.
* **Mean Absolute Error (MAE)**: **0.228 points** on the raw 1.0–5.0 scale, with **100.0% of all ratings within 0.5 points** of human ground truth.
* **Cohen's $\kappa$ Handling**: Cohen's Kappa is undefined/NaN on dimensions where both evaluators assign uniform high scores (Groundedness, Actionability), which is represented as `null` in JSON and reported strictly as `N/A` at the presentation layer rather than using artificial 1.0 substitutions. On binary Escalation Appropriateness, agreement is $\kappa = 1.000$; Brand Voice agreement is $\kappa = 1.000$; Overall is $\kappa = 0.215$.
* **Groundedness Score Distribution**: Both the human evaluator and the LLM judge scored candidate replies high on grounding (mean ratings 4.60 and 4.95 respectively, with 100% within 0.5 points and MAE = 0.350), because candidate responses consistently cite valid official Apple support URLs and verbatim Apple steps. This low score variance naturally produces a modest Pearson correlation ($r=0.167$), while absolute error confirms close agreement.
* **Input Alignment Verification**: 100% of evaluated pairs match candidate SHA256 input hashes (`item_id`, `query`, `gold_intent`, `gold_escalation`, `candidate_reply`, `candidate_escalation`, `rubric_version`). These cryptographic hashes verify identical evaluation inputs between human and LLM scoring; they prove input alignment only, not evaluator identity.

---

## Section 6: What We Would Do Next with One More Week

1. **Tri-State Copilot Routing**: Implement High Confidence (>0.85) -> Auto-Reply; Medium Confidence (0.55–0.85) -> One-Click Draft in Hiver inbox for human agent review; Low Confidence -> Direct Escalation.
2. **Stateful Multi-Turn Dialog Trees**: Track user conversation state across multiple tweets and ask structured disambiguating questions when symptoms are vague.
3. **Mock CRM Tool Calling**: Connect agent to live Apple System Status API endpoints and AppleCare warranty entitlement lookups before generating replies.
4. **Active Learning Loop**: Automatically route low-confidence customer queries to human reviewers daily to continuously expand the golden benchmark.
