# AI Customer Support Agent for @AppleSupport: Comprehensive Technical Report

**Author**: Hiver SDE Intern Candidate  
**Date**: September 2026  
**Target Brand**: Apple Support (`@AppleSupport` on Twitter/X)  
**Dataset**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`, threaded by `TNE-AI`)  
**Holdout Evaluation Set**: 200 Hand-Curated, Stratified Gold Examples (`data/golden_eval_set.json`)  
**Compiled PDF**: [REPORT.pdf](REPORT.pdf) (Strictly <= 6 Pages, Currently 3 Pages)

---

## Executive Summary

Customer support on social media is high-stakes, real-time, and public. For a brand like Apple, an AI support agent must do more than answer questions: it must maintain customer trust, protect user security and data privacy, comply strictly with character limits, and know with certainty **when not to answer**.

This report documents the design, implementation, and empirical evaluation of an **Evaluation-Focused AI Support Agent Prototype with Calibrated Safety Guardrails** for `@AppleSupport`. We evaluate our system against two baselines (a Trivial Majority-Rule Baseline and a Classical Statistical Machine Learning Baseline) on a 200-sample hand-labelled Golden Evaluation Set using a strictly thread-disjoint split (zero conversation ID overlap between 600 train threads, 1,000 KB threads, and 200 holdout gold threads). 

The proposed agent achieves:
- **64.5% Out-of-Sample Intent Accuracy** across a 7-class domain taxonomy (vs. 35.5% Trivial and 54.5% Simple Baseline).
- **70.7% Escalation Recall** on safety-critical interactions (vs. **0.0%** for Trivial Baseline and **1.7%** for Simple Baseline).
- **79.5% Official Apple Domain Link Validity** and **73.0% Intent-Link Relevance Rate** (vs. 0.0% for baselines).
- **100.0% Twitter Character Limit Compliance** (<280 chars).
- Mean Heuristic Rubric score of **4.67 / 5.00** (vs. 4.35 for Trivial and 4.17 for Simple Baseline).
- Paired Human vs. LLM-as-a-Judge Study ($N=50$): **Pearson $r = 0.964$**, **Spearman $\rho = 0.986$**, **MAE = 0.108 points**, and **Cohen's $\kappa = 0.733$**.
- The full evaluation suite reproduces completely offline on standard CPU in **7.8 seconds**, easily satisfying the <15-minute reproduction requirement with zero external API dependencies.

---

## Section 1: Problem Framing & Scope Boundaries

### 1.1 What "Good" Means for @AppleSupport
On Twitter, `@AppleSupport` handles hundreds of thousands of customer inquiries monthly spanning hardware, software, and cloud services. Through analysis of the historical corpus, "good" support for Apple is characterized by four non-negotiable operational principles:

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
   - Escalation: **Always predict `AUTO_HANDLE`** (majority triage class, reflecting 71% of incoming traffic).
   - Reply: Static generic canned macro reply ("Thanks for reaching out! We'd love to help. Please DM us your device model and iOS version so we can look into this.").
2. **Simple Baseline**:
   - Intent: Multinomial Naive Bayes trained on word count vectors.
   - Escalation: Simple keyword-matching rules (looking for words like "refund", "human", "agent", "sue", "manager").
   - Reply: 1-Nearest-Neighbor historical reply retrieval.
3. **Proposed AI Agent**:
   - Intent: Calibrated Logistic Regression with TF-IDF features, subword n-grams, and class-balanced weights.
   - Grounded RAG: Semantic TF-IDF vector index over 1,000 historical brand resolutions, extracting concrete resolution clauses from historical replies with canonical Apple domain whitelisting.
   - Escalation Engine: Asymmetric safety policy rules + confidence thresholds with explicit stated reasons.
   - Reply Generator: Brand-conditioned reply drafting enforcing Twitter <280-char constraints with structured `grounded_in` evidence snippets.

### 2.2 Benchmark Results Table (Zero-Leakage Thread-Disjoint Split)

| Metric Dimension | Trivial Baseline (Always Auto-Handle) | Simple Baseline (Naive Bayes + 1-NN) | Proposed AI Agent (RAG + Policy) | Real-World Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Classification Accuracy** | 35.5% | 54.5% | **64.5%** | Out-of-sample generalization across 7 domain intents |
| **Intent Macro F1** | 0.075 | 0.262 | **0.535** | Superior balance on under-represented intents |
| **Escalation Decision Accuracy** | 71.0% | 71.0% | **76.5%** | Higher overall triage accuracy |
| **Escalation Recall (Safety-Critical)** | **0.0%** | **1.7%** | **70.7%** | Baselines miss 98.3% to 100% of cases needing humans |
| **Escalation Precision** | 0.0% | 50.0% | **57.8%** | Balanced triage efficiency preventing agent overload |
| **False Escalation Rate (Lower=Better)**| 0.0% | 0.7% | **21.1%** | Trade-off: accepts ~21% false alarms to catch critical risks |
| **SacreBLEU Score** | 0.2 | 0.3 | **4.7** | Substantial improvement over generic baselines |
| **Twitter Char Limit Compliance (<280)** | 100.0% | 92.5% | **100.0%** | Zero tweet truncation or broken URL artifacts |
| **Official Domain Link Validity** | 0.0% | 0.0% | **79.5%** | Provides verified official Apple URLs vs stale redirects |
| **Intent-Link Relevance Rate** | 0.0% | 0.0% | **73.0%** | Canonical URL matches classified problem domain |
| **Heuristic: Groundedness (1–5)** | 4.20 | 4.05 | **4.78** | Factual grounding in historical troubleshooting steps |
| **Heuristic: Brand Voice & Empathy (1–5)** | 5.00 | 4.30 | **4.61** | Professional Apple tone without sounding robotic |
| **Heuristic: Actionability (1–5)** | 4.20 | 4.33 | **4.81** | High practical utility and clear next actions |
| **Heuristic: Escalation Appropriateness (1–5)**| 4.00 | 4.01 | **4.46** | Safe triage decisions aligned with enterprise risk policy |
| **Heuristic: Overall Quality Score (1–5)** | 4.35 | 4.17 | **4.67** | Clear superiority across combined holistic criteria |

---

## Section 3: Failure Mode Analysis (Top 5 Real Observed Failures)

A transparent post-mortem of our model's errors is essential to earning organizational trust. All 5 cases are genuine observed errors from `evaluation/error_analysis.json`:

### Failure Mode 1: Sarcasm and Ambiguous Frustration [GOLD_002]
* **Customer Query**:
  > *"It's been nearly two weeks and I've yet to get LTE on my Watch () working. Think I paid a premium for a spec of red paint. Yes and I talked to Apple Care a few days ago."*
* **Observed Agent Output**: Predicted Intent = `IOS_SOFTWARE_UPDATE` (Confidence: 0.31). Correctly escalated to human via `POLICY_LOW_MODEL_CONFIDENCE`.
* **Gold Ground Truth**: Intent = `CUSTOMER_FEEDBACK_COMPLAINT`, Escalation = `ESCALATE`.
* **Root Cause & Mitigation**: Sarcastic phrasing ("paid a premium for a spec of red paint") and mention of LTE Watch connectivity confused intent classification. However, confidence thresholding safely caught the ambiguity and triggered escalation. Mitigation: Multi-turn sentiment tracking and keyword flagging for unresolved multi-day periods ("nearly two weeks", "days ago").

### Failure Mode 2: Multilingual Language Routing Miss [GOLD_004]
* **Customer Query**:
  > *"Mira que me gusta vuestra actualización pero me va como el culo ahora, cuando queráis lo solucionáis."*
* **Observed Agent Output**: Predicted Intent = `OUT_OF_SCOPE_OTHER` (Confidence: 0.87), Decision = `AUTO_HANDLE` (Sent generic English support link).
* **Gold Ground Truth**: Intent = `OUT_OF_SCOPE_OTHER`, Escalation = `ESCALATE` (Non-English localization routing).
* **Root Cause & Mitigation**: The policy engine failed to trigger `POLICY_LANGUAGE_LOCALIZATION` on informal European Spanish, producing a generic English auto-reply. Mitigation: Integrate fastText language identification at the ingestion gateway before intent classification.

### Failure Mode 3: False-Positive Escalation on Repeated Troubleshooting Phrasing [GOLD_008]
* **Customer Query**:
  > *"I've had to do it multiple times when I reset my device and how do I change trusted device from old phone to new phone?"*
* **Observed Agent Output**: Decision = `ESCALATE` (Triggered via `POLICY_REPEATED_UNRESOLVED_FAILURE`).
* **Gold Ground Truth**: Decision = `AUTO_HANDLE` (Routine device pairing how-to).
* **Root Cause & Mitigation**: The phrase "multiple times" triggered the repeated failure heuristic on a standard UI how-to query. Mitigation: Condition repeated-failure rules on negative emotional sentiment tokens rather than raw occurrence words.

### Failure Mode 4: Cross-Platform System Hang Misclassification [GOLD_019]
* **Customer Query**:
  > *"Cool new feature in macOS High Sierra, it knows you've been working too hard and freezes the screen, but not the mouse. Can't get to the force quit menu, can't switch apps, can't get to the login screen. The only recourse is to hold power button down."*
* **Observed Agent Output**: Intent = `DEVICE_SETUP_AND_USAGE` (Confidence: 0.59), replied with iPhone user guide link (`support.apple.com/guide/iphone`).
* **Gold Ground Truth**: Intent = `IOS_SOFTWARE_UPDATE` (macOS system freeze), Decision = `AUTO_HANDLE`.
* **Root Cause & Mitigation**: Training set priors are dominated by iPhone iOS queries, causing Mac desktop queries to latch onto mobile setup guides. Mitigation: Device-type entity extraction filter prior to retrieval to restrict KB scope to Mac-specific articles.

### Failure Mode 5: Out-of-Warranty Hardware Defect Under-Escalation [GOLD_047]
* **Customer Query**:
  > *"Hi Apple! My early 2015 Macbook Pro Retina has glares on it. Since I have no Apple Care, will I be able to replace the anti-reflective coating?"*
* **Observed Agent Output**: Intent = `OUT_OF_SCOPE_OTHER` (Confidence: 0.33), Decision = `AUTO_HANDLE`.
* **Gold Ground Truth**: Intent = `BATTERY_AND_HARDWARE`, Escalation = `ESCALATE`.
* **Root Cause & Mitigation**: The customer inquired about the known "Staingate" anti-reflective coating quality program. Lacking specialized hardware keywords, the agent sent a generic link instead of scheduling a Genius Bar hardware evaluation. Mitigation: Expand hardware taxonomy dictionary with known quality-program keywords ("coating", "delamination", "display glare").

---

## Section 4: "What is Misleading About My Headline Number?" (Mandatory Section)

A candidate who blindly presents high numbers without understanding their operational reality cannot be trusted in production. Here is our rigorous critique:

### 1. The Deception of Raw Accuracy in Imbalanced Triage
Our Escalation Accuracy is **76.5%**, which is only slightly above the baselines (71.0%). Looking at accuracy alone would suggest that the AI agent adds negligible value over guessing `AUTO_HANDLE` for everything!
However, this exposes the central failure of raw accuracy on imbalanced distributions:
- Trivial Baseline Escalation Recall: **0.0%** (catches 0 out of 58 human escalations).
- Simple Baseline Escalation Recall: **1.7%** (catches 1 out of 58 human escalations).
- Proposed AI Agent Escalation Recall: **70.7%** (catches 41 out of 58 human escalations).
Accuracy is an actively misleading metric in customer support triage.

### 2. The Offline-to-Online Dynamic Gap
In our offline benchmark, the agent provides a grounded reply and link (`support.apple.com/HT204204`), earning a 4.81 / 5.0 for Actionability. But in reality, customer support is dynamic. If the customer clicks the link, fails to understand step 2, and tweets back: *"That didn't work. Now what?"*, our offline evaluation awards full credit for an interaction that actually resulted in zero first-contact resolution.

### 3. Intent Accuracy is Bound by Domain Taxonomy Framing
Our out-of-sample intent accuracy is 64.5% across 7 coarse categories. If we evaluated on fine-grained categories (e.g. 50+ intents) or unconstrained open-domain inputs, performance would drop significantly. The metric measures consistency within our defined taxonomy, not open-ended comprehension.

### 4. Judge Heuristic Leniency Bias
Our automated rubric awarded a mean score of 4.67 / 5.0. However, automated rubrics possess an inherent leniency bias toward syntactically clean, polite text containing domain keywords ("Settings", "Apple Support", "DM us"). An authoritative-sounding reply that provides a deprecated iOS 10 step could still score high if not caught by manual human spot-checking.

---

## Section 5: Human-Judge Inter-Rater Reliability (N=50 Paired Frozen Outputs)

To validate our automated evaluator, we conducted a blind inter-rater reliability study comparing an LLM judge and human expert ratings on the exact same 50 frozen agent outputs:

* **Pearson Correlation ($r$)**: **0.964** on Overall Rubric Score, indicating near-perfect linear tracking of human scoring.
* **Spearman Rank Correlation ($\rho$)**: **0.986**, demonstrating consistent ordinal ranking of response quality.
* **Mean Absolute Error (MAE)**: **0.108 points** on the raw 1.0–5.0 scale, with **100.0% of all ratings within 0.5 points** of the human ground truth.
* **Cohen's Kappa ($\kappa$)**: $\kappa = 0.733$ on binned quality tiers, confirming strong agreement well beyond chance.

---

## Section 6: What We Would Do Next with One More Week

1. **Tri-State Copilot Routing**: Implement High Confidence (>0.85) -> Auto-Reply; Medium Confidence (0.55–0.85) -> One-Click Draft in Hiver inbox for human agent review; Low Confidence -> Direct Escalation.
2. **Stateful Multi-Turn Dialog Trees**: Track user conversation state across multiple tweets and ask structured disambiguating questions when symptoms are vague.
3. **Mock CRM Tool Calling**: Connect agent to live Apple System Status API endpoints and AppleCare warranty entitlement lookups before generating replies.
4. **Active Learning Loop**: Automatically route low-confidence customer queries to human reviewers daily to continuously expand the golden benchmark.
