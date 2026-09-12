# AI Customer Support Agent for @AppleSupport

> **Hiver SDE Intern Take-Home Assignment**: An end-to-end, evaluation-focused AI Customer Support Agent prototype for `@AppleSupport` built on real Twitter customer support data. Classifies incoming customer messages into domain intents, drafts grounded replies using historical brand resolutions, and renders calibrated, auditable auto-handle vs. human escalation decisions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/codebyArya-bit/hiver-sde-takehome-apple-support-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/codebyArya-bit/hiver-sde-takehome-apple-support-agent/actions)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(19%2F19)-brightgreen.svg)]()
[![Reproduction Time](https://img.shields.io/badge/Reproduction%20Time-%3C10%20Seconds-success.svg)]()
[![Data Leakage](https://img.shields.io/badge/Thread%20Leakage-0%20Overlaps-success.svg)]()
[![Brand](https://img.shields.io/badge/Brand-%40AppleSupport-black.svg)]()

---

## 💡 Quick Examples: How the Agent Works

To understand the system in 30 seconds, here are two end-to-end execution traces demonstrating its dual mechanisms:

### Example 1: Intent-Independent Safety Escalation (Security Boundary)
When an account compromise or security risk occurs, the safety layer intercepts critical security keywords (`hacked`, `compromised`, `unauthorized`) via **`POLICY_SECURITY_CREDENTIALS`** (fired specifically on the keyword `hacked`) and enforces immediate escalation with official recovery routing, completely independent of model intent confidence.

**Incoming Customer Tweet:**
> *"SOME ASSHOLE HACKED INTO MY ITUNES AND BOUGHT A BUNCH OF EMO MUSIC ON MY ACCOUNT PLEASE HELP ME CANCEL THIS"*

**Agent Pipeline Execution:**
```json
{
  "intent": "OUT_OF_SCOPE_OTHER",
  "intent_confidence": 0.49,
  "escalation_decision": "ESCALATE",
  "escalation_reason": "Account recovery, credential resets, and 2FA authentication cannot be safely resolved over public social channels.",
  "policy_triggered": "POLICY_SECURITY_CREDENTIALS",
  "draft_reply": "We'd love to look into this with you directly. Please send us a DM with your device model, current iOS version, and any details so we can assist: https://twitter.com/messages/compose",
  "used_evidence": []
}
```

### Example 2: Grounded Historical RAG (Actionable Resolution Extraction)
For routine technical troubleshooting, the agent uses Sparse TF-IDF retrieval over 1,000 historical `@AppleSupport` resolutions, extracts concrete imperative troubleshooting clauses from authentic past brand replies, and incorporates them directly into the generated response:

**Incoming Customer Tweet:**
> *"This #iphone update is crap! My screen brightness won't stay where I have it set, auto adjusts on its own when I have it set not to"*

**Agent Pipeline Execution:**
```json
{
  "intent": "IOS_SOFTWARE_UPDATE",
  "intent_confidence": 0.55,
  "escalation_decision": "AUTO_HANDLE",
  "escalation_reason": "Standard iOS software troubleshooting steps (restart, storage check, supplemental update) apply.",
  "policy_triggered": "NONE_AUTO_HANDLED",
  "draft_reply": "We're happy to help! Tap Settings > Display & Brightness. Check out more troubleshooting steps here: https://support.apple.com/HT204204",
  "used_evidence": [
    {
      "conversation_id": "0cca95c422a98097f0f9d199ec771d02",
      "similarity": 0.1464,
      "resolution_snippet": "We'll be glad to assist you. Is your iPhone set to auto-lock? Tap Settings > Display & Brightness. What is Aut"
    }
  ],
  "extracted_action_used": true
}
```

---

## 📊 Headline Benchmark Results

Evaluated across the **200 hand-audited holdout Golden Evaluation Set** (`data/golden_eval_set.json`). The 200-item gold holdout is thread-disjoint from both the 600 training examples and the 1,000-item historical retrieval KB. The 600 classifier-training conversations are included within the 1,000-item historical KB:

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

*Benchmark execution runtime: **~6.1 seconds on standard CPU** (reproduces offline with zero API keys).*

### Candidate Human Annotator vs. LLM-as-a-Judge Agreement ($N=50$)
The candidate manually scored 50 frozen outputs using the four-axis rubric without viewing the LLM ratings, then compared the two rating sets:
- **Pearson Correlation ($r$)**: **0.940** (Strong linear tracking)
- **Spearman Rank Correlation ($\rho$)**: **0.785** (Consistent ordinal quality ranking)
- **Mean Absolute Error (MAE)**: **0.228 points** on a 1–5 scale (**100.0% within 0.5 points**)
- **Cohen's $\kappa$ Handling**: Cohen's Kappa is undefined/NaN on dimensions where both evaluators assign uniform high scores (Groundedness, Actionability), which is represented as `null` in JSON and reported strictly as `N/A` at the presentation layer rather than using artificial 1.0 substitutions. On binary Escalation Appropriateness, agreement is $\kappa = 1.000$; Brand Voice agreement is $\kappa = 1.000$; Overall is $\kappa = 0.215$.
- **Groundedness Agreement**: 100% of ratings within 0.5 points with MAE = 0.350. Restricted score variance (both raters awarding >4.0 due to verified Apple URLs) accounts for lower linear variance ($r=0.167$) while absolute agreement remains high.
- **Input Alignment Verification**: 100% of evaluated pairs match candidate SHA256 input hashes (`item_id`, `query`, `gold_intent`, `gold_escalation`, `candidate_reply`, `candidate_escalation`, `rubric_version`). These cryptographic hashes verify identical evaluation inputs between human and LLM scoring; they prove input alignment only, not evaluator identity.

---

## 📁 Key Deliverables & Documentation

- 📄 **[Technical Report (PDF, <= 6 Pages)](REPORT.pdf)**: Print-ready publication PDF strictly adhering to the $\le 6$ page limit (currently 3 pages, enforced programmatically).
- 📝 **[Full Markdown Report](REPORT.md)**: Problem framing, non-goals, baselines, failure modes with authentic `GOLD_xxx` IDs, the mandatory critique *"What is misleading about my headline number?"*, and future work.
- 📋 **[Engineering Decision Log](DECISION_LOG.md)**: 15 non-obvious engineering decisions and trade-offs.
- 🏷️ **[Golden Set Annotation Guide](data/GOLD_ANNOTATION_GUIDE.md)** & **[CSV Export](data/gold_annotations.csv)**: Complete human annotation guide and raw CSV audit provenance for all 200 items.
- 🛡️ **[Adversarial Stress Test Suite](data/adversarial_stress_test.json)**: 12 authentic synthetic stress test cases (jailbreak, thermal swelling, account takeover, PII disclosure, legal threat, non-English).
- 📚 **[Attributions & Citations](ATTRIBUTIONS.md)**: Citations for datasets (Kaggle CC BY-NC-SA 4.0), research papers, open-source libraries, and Apple documentation.

---

## 🏗️ Architecture Overview

```
+---------------------------------------------------------------------------------+
|                               Incoming Customer Tweet                           |
+---------------------------------------------------------------------------------+
                                         |
                                         v
               +--------------------------------------------------+
               | 1. Intent Classification Engine                  |
               | (Taxonomy: Battery/Hardware, OS/Software Bug,     |
               |  Apple ID/iCloud, Billing/Store, How-To/Setup,   |
               |  Feedback/Rant, Out-of-Scope/Spam)               |
               | - Calibrated Logistic Regression (ECE = 0.040)   |
               +--------------------------------------------------+
                                         |
                        +----------------+----------------+
                        |                                 |
                        v                                 v
   +---------------------------------------+   +------------------------------------+
   | 2. Grounded Resolution Engine (RAG)  |   | 3. Escalation Decision Engine      |
   | - Sparse TF-IDF retrieval over        |   | - Intent-Independent Safety Policy |
   |   1,000 historical resolutions        |   | - Calibrated Confidence Scorer     |
   | - Actionable resolution extraction    |   | - Decision: Auto-Handle vs Escalate|
   | - Canonical Apple URL whitelist       |   | - Explicit Stated Reason Generator |
   | - Twitter <280-char compliance        |   | - Asymmetric Recall Priority (F2)  |
   +---------------------------------------+   +------------------------------------+
                        \                                 /
                         \                               /
                          v                             v
                   +-------------------------------------------+
                   | Final Agent Output:                       |
                   | - Intent + Calibrated Confidence          |
                   | - Grounded Draft Reply (<280 chars)       |
                   | - Routing Decision (Auto/Escalate)        |
                   | - Stated Escalation Reason                |
                   | - Used & Retrieved Evidence Citations     |
                   +-------------------------------------------+
```

---

## 🚀 Clean-Clone Quickstart & Reproduction (< 15 Minutes)

The entire pipeline is self-contained, lightweight, and requires **zero external API keys** for offline CPU reproduction.

### 1. Clone & Setup Environment
```bash
git clone https://github.com/codebyArya-bit/hiver-sde-takehome-apple-support-agent.git
cd hiver-sde-takehome-apple-support-agent

# Create virtual environment
python -m venv .venv

# Activate environment:
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
# .venv\Scripts\Activate.ps1

# Install dependencies (< 30 seconds)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# (Optional: exact pinned lockfile: pip install -r requirements-lock.txt)
```

### 2. Verify Data Integrity & Split Constraints
```bash
python scripts/verify_splits.py
```
*Verifies 0 thread overlaps between Train (600) vs Gold (200) and KB (1000) vs Gold (200), confirms Train is included in KB (600/600), and validates single-turn evaluation constraints.*

### 3. Run the Headline Benchmark Harness
```bash
python evaluation/run_evaluation.py
```
*Executes all three agents on the 200 Golden Evaluation examples, generates comparison tables and human-judge agreement stats, and outputs `evaluation/benchmark_results.json` and `evaluation/error_analysis.json` in ~6 seconds.*

### 4. Run Automated Test Suite
```bash
python -m pytest tests/test_pipeline.py -v
```
*Executes 19 comprehensive unit & regression tests (leakage, single-turn evaluation integrity, cryptographic hash verification, calibration, all 12 adversarial stress cases, hard safety independence, Cohen's Kappa, judge provenance, benchmark consistency).*

### 5. Recompile the Publication PDF Report
```bash
python scripts/generate_pdf_report.py
```
*Compiles `REPORT.pdf` dynamically from benchmark results with programmatic $\le 6$ page enforcement.*

### 6. Interactive Live Agent Demo
Test the agent interactively with any custom customer tweet:
```bash
python -c "
from src.agent import AppleSupportAgent
agent = AppleSupportAgent().initialize()
result = agent.process_message('My iPhone battery drops from 80% to 10% in 15 minutes and gets burning hot!')
import json
print(json.dumps(result, indent=2))
"
```
