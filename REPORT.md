# AI Customer Support Agent for @AppleSupport: Comprehensive Technical Report

**Author**: Hiver SDE Intern Candidate  
**Date**: September 2026  
**Target Brand**: Apple Support (`@AppleSupport` on Twitter/X)  
**Dataset**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`, threaded by `TNE-AI`)  
**Holdout Evaluation Set**: 200 Hand-Curated, Stratified Gold Examples (`data/golden_eval_set.json`)  
**Compiled PDF**: [REPORT.pdf](REPORT.pdf) (Strictly <= 6 Pages)

---

## Executive Summary

Customer support on social media is high-stakes, real-time, and public. For a brand like Apple, an AI support agent must do more than answer questions: it must maintain customer trust, protect user security and data privacy, comply strictly with character limits, and know with certainty **when not to answer**.

This report documents the design, implementation, and empirical evaluation of an **Evaluation-Focused AI Support Agent Prototype with Calibrated Safety Guardrails** for `@AppleSupport`. We evaluate our system against two baselines (a Trivial Majority-Rule Baseline and a Classical Statistical Machine Learning Baseline) on a 200-sample hand-labelled Golden Evaluation Set using a strictly thread-disjoint split (zero conversation ID overlap). 

The proposed agent achieves:
- **69.5% Out-of-Sample Intent Accuracy** across a 7-class domain taxonomy (vs. 39.0% Trivial and 58.0% Simple Baseline).
- **63.8% Escalation Recall** on safety-critical interactions (vs. **0.0%** for Trivial Baseline and **3.5%** for Simple Baseline).
- **79.5% Official Apple Domain Link Validity** and **72.6% Intent-Link Relevance Rate** (vs. 0.0% for baselines).
- **100.0% Twitter Character Limit Compliance** (<280 chars).
- Mean LLM-as-a-judge quality score of **4.63 / 5.00** (vs. 4.14 for Trivial and 4.16 for Simple Baseline).
- The full evaluation suite reproduces completely offline on standard CPU in **6.2 seconds**, easily satisfying the <15-minute reproduction requirement with zero external API dependencies.

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
   - Intent: Hybrid calibrated classifier combining TF-IDF representations with Bayesian domain pattern priors.
   - Grounded RAG: Semantic vector index over 1,000 historical brand resolutions with canonical Apple domain whitelisting.
   - Escalation Engine: Asymmetric safety policy rules + confidence thresholds with explicit stated reasons.
   - Reply Generator: Brand-conditioned reply drafting enforcing Twitter <280-char constraints.

### 2.2 Benchmark Results Table (Zero-Leakage Thread-Disjoint Split)

| Metric Dimension | Trivial Baseline (Always Auto-Handle) | Simple Baseline (Naive Bayes + 1-NN) | Proposed AI Agent (RAG + Policy) | Real-World Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Classification Accuracy** | 39.0% | 58.0% | **69.5%** | Out-of-sample generalization across 7 domain intents |
| **Intent Macro F1** | 0.080 | 0.283 | **0.527** | Superior balance on under-represented intents |
| **Escalation Decision Accuracy** | 71.0% | 71.5% | **71.5%** | Comparable overall accuracy, but vastly different recall |
| **Escalation Recall (Safety-Critical)** | **0.0%** | **3.5%** | **63.8%** | Baselines miss 96.5% to 100% of cases needing humans |
| **Escalation Precision** | 0.0% | 66.7% | **50.7%** | Balanced triage efficiency |
| **False Escalation Rate (Lower=Better)**| 0.0% | 0.7% | **25.4%** | Trade-off: accepts ~25% false alarms to catch critical risks |
| **Twitter Char Limit Compliance (<280)** | 100.0% | 93.0% | **100.0%** | Zero tweet truncation or broken URL artifacts |
| **Official Domain Link Validity** | 0.0% | 0.0% | **79.5%** | Provides verified official Apple URLs vs stale t.co redirects |
| **Intent-Link Relevance Rate** | 0.0% | 0.0% | **72.6%** | Canonical URL matches classified problem domain |
| **Judge: Groundedness (1–5)** | 4.20 | 4.06 | **4.79** | Factual grounding in real troubleshooting steps |
| **Judge: Brand Voice & Empathy (1–5)** | 5.00 | 4.31 | **4.58** | Professional Apple tone without sounding robotic |
| **Judge: Actionability (1–5)** | 3.40 | 4.31 | **4.81** | High practical utility and clear next actions |
| **Judge: Escalation Appropriateness (1–5)**| 3.95 | 3.98 | **4.34** | Safe triage decisions aligned with enterprise risk policy |
| **Judge: Overall Quality Score (1–5)** | 4.14 | 4.16 | **4.63** | Clear superiority across combined holistic criteria |

---

## Section 3: Failure Mode Analysis (Top 5 Failure Modes)

A transparent post-mortem of our model's errors is essential to earning organizational trust:

### Failure Mode 1: Sarcasm and Idiomatic Frustration
* **Real Query Example**:
  > *"It's been nearly two weeks and I've yet to get LTE on my Watch working. Think I paid a premium for a spec of red paint."*
* **Model Output**: Intent = `DEVICE_SETUP_AND_USAGE`, Decision = `AUTO_HANDLE` (Advised checking Cellular settings and watchOS update).
* **Gold Ground Truth**: Intent = `CUSTOMER_FEEDBACK_COMPLAINT`, Decision = `ESCALATE`.
* **Root Cause Hypothesis**: The customer does not use explicit profanity or standard escalation keywords like "sue", "manager", or "fraud". The frustration is communicated through sarcasm ("paid a premium for a spec of red paint"). Shallow embeddings interpret "LTE on my Watch working" as a routine connectivity setup question rather than an escalated customer churn risk.
* **Mitigation**: Deploy a dedicated sentiment and sarcasm detection sidecar trained on customer support frustration corpora, or flag any inquiry mentioning an unresolved timeframe exceeding 7 days ("two weeks", "days now") as an automatic escalation candidate.

### Failure Mode 2: Multi-Intent / Compound Inquiries
* **Real Query Example**:
  > *"I updated to iOS 11.0.2 and my phone is freezing constantly. Also I noticed a mysterious $9.99 charge from iTunes on my bank statement!"*
* **Model Output**: Intent = `IOS_SOFTWARE_UPDATE`, Decision = `AUTO_HANDLE` (Advised force restart and storage check).
* **Gold Ground Truth**: Intent = `APP_STORE_AND_BILLING`, Decision = `ESCALATE`.
* **Root Cause Hypothesis**: The single-label classifier picked up strong software signals ("updated", "iOS 11.0.2", "freezing") which appeared first in the sentence. However, the secondary clause contained a financial billing dispute ("mysterious $9.99 charge"). Because the architecture assigned a single label, the billing issue was overshadowed, leading to an incorrect `AUTO_HANDLE` decision.
* **Mitigation**: Implement multi-label intent classification with an asymmetric triage hierarchy: if ANY detected sub-intent belongs to a safety-critical category (`APP_STORE_AND_BILLING` or `APPLE_ID_AND_ICLOUD`), the entire conversation inherits the escalation requirements of the highest-risk sub-intent.

### Failure Mode 3: Hardware Thermal Safety Under-Specification
* **Real Query Example**:
  > *"My phone gets warm when charging on iOS 11."* vs *"My phone got so hot it smelled like smoke and the battery is bulging."*
* **Model Output**: Intent = `BATTERY_AND_HARDWARE` for both.
* **Boundary Fragility**: When customers use ambiguous language such as *"My phone is burning up"*, this can idiomatically mean normal processor heat during gaming or an actual lithium-ion thermal runaway event. When the model misclassifies colloquial heat ("burning up") as a standard battery diagnostic issue, it risks auto-handling a hazardous device.
* **Mitigation**: Introduce a safety clarification prompt: when heat-related terms are detected without explicit damage keywords, immediately prompt: *"If your device is uncomfortably hot to touch, swollen, or showing signs of damage, please disconnect power immediately and DM us. If it is only warming up during gaming or updates, see our battery guide..."*

### Failure Mode 4: False Escalation on Standard FAQs with Negative Sentiment
* **Real Query Example**:
  > *"I hate this stupid update! Where did the shuffle button go in Apple Music? It is impossible to find!"*
* **Model Output**: Intent = `CUSTOMER_FEEDBACK_COMPLAINT`, Decision = `ESCALATE` (Triggered on "hate", "stupid", "impossible").
* **Gold Ground Truth**: Intent = `DEVICE_SETUP_AND_USAGE`, Decision = `AUTO_HANDLE`.
* **Root Cause Hypothesis**: The customer's emotional venting triggered the escalation policy for negative sentiment, despite the underlying technical problem being a trivial 5-second UI navigation question.
* **Mitigation**: Decouple sentiment from technical resolvability. If an inquiry has a deterministic UI answer (e.g. "where is the shuffle button"), answer the technical question directly while adopting an empathetic, de-escalating tone, reserving human escalation for when the issue remains unresolved.

### Failure Mode 5: Language Drift and Regional Slang
* **Real Query Example**:
  > *"Awrite av got a problem with my iPhone i was just wondering how you get rid of the padlock icon at the top right of the screen?"*
* **Model Output**: Intent = `DEVICE_SETUP_AND_USAGE`, Decision = `AUTO_HANDLE` (Correct).
* **Observed Vulnerability**: Regional colloquialisms ("Awrite av got") distort subword segmenters, causing lower intent confidence scores (< 0.40) that trigger `POLICY_LOW_MODEL_CONFIDENCE` and cause unnecessary human escalations.
* **Mitigation**: Add a lightweight text normalization layer that standardizes common regional idioms and phonetic slang before feature extraction.

---

## Section 4: "What is Misleading About My Headline Number?" (Mandatory Section)

A candidate who blindly presents high numbers without understanding their operational reality cannot be trusted in production. Here is our rigorous critique:

### 1. The Deception of Raw Accuracy in Imbalanced Triage
Our Escalation Accuracy is **71.5%**, which is identical to the Simple Baseline (71.5%) and almost identical to the Trivial Baseline (71.0%). Looking at accuracy alone would suggest that the AI agent adds zero value over guessing `AUTO_HANDLE` for everything!
However, this exposes the central failure of raw accuracy on imbalanced distributions:
- Trivial Baseline Escalation Recall: **0.0%** (catches 0 out of 58 human escalations).
- Simple Baseline Escalation Recall: **3.5%** (catches 2 out of 58 human escalations).
- Proposed AI Agent Escalation Recall: **63.8%** (catches 37 out of 58 human escalations).
Accuracy is an actively misleading metric in customer support triage.

### 2. The Offline-to-Online Dynamic Gap
In our offline benchmark, the agent provides a grounded reply and link (`support.apple.com/HT204204`), earning a 4.81 / 5.0 for Actionability. But in reality, customer support is dynamic. If the customer clicks the link, fails to understand step 2, and tweets back: *"That didn't work. Now what?"*, our offline evaluation awards full credit for an interaction that actually resulted in zero first-contact resolution.

### 3. Intent Accuracy is Bound by Domain Taxonomy Framing
Our out-of-sample intent accuracy is 69.5% across 7 coarse categories. If we evaluated on fine-grained categories (e.g. 50+ intents) or unconstrained open-domain inputs, performance would drop significantly. The metric measures consistency within our defined taxonomy, not open-ended comprehension.

### 4. LLM Judge Leniency Bias
Our automated judge awarded a mean score of 4.63 / 5.0. However, automated rubrics possess an inherent leniency bias toward syntactically clean, polite text containing domain keywords ("Settings", "Apple Support", "DM us"). An authoritative-sounding reply that provides a deprecated iOS 10 step could still score 4/5 from an automated judge if not caught by manual human spot-checking.

---

## Section 5: Human-Judge Inter-Rater Reliability (N=200)

To establish trust in our automated evaluator, we measured alignment between judge ratings and human annotations across all 200 items:

* **Pearson Correlation ($r$)**: **0.343** on Overall Score and **0.392** on Escalation Appropriateness, demonstrating moderate linear alignment.
* **Mean Absolute Error (MAE)**: **0.290 points** on the raw 1.0–5.0 scale, with **90.0% of all ratings within 0.5 points** of the human ground truth.
* **Cohen's Kappa ($\kappa$)**: $\kappa = 0.222$ on binned quality tiers, confirming agreement above random chance without artificial score inflation.

---

## Section 6: What We Would Do Next with One More Week

1. **Tri-State Copilot Routing**: Implement High Confidence (>0.85) -> Auto-Reply; Medium Confidence (0.55–0.85) -> One-Click Draft in Hiver inbox for human agent review; Low Confidence -> Direct Escalation.
2. **Stateful Multi-Turn Dialog Trees**: Track user conversation state and ask structured disambiguating questions when symptoms are vague.
3. **Mock CRM Tool Calling**: Connect agent to live Apple System Status API endpoints and AppleCare warranty entitlement lookups before generating replies.
4. **Active Learning Loop**: Automatically route low-confidence customer queries to human reviewers daily to continuously expand the golden benchmark.
