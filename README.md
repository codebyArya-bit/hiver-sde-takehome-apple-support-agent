# AI Customer Support Agent for @AppleSupport

> **Hiver SDE Intern Take-Home Assignment**: An end-to-end, evaluation-focused AI Customer Support Agent prototype for `@AppleSupport` built on real Twitter customer support data. Classifies incoming customer messages into domain intents, drafts grounded replies using historical brand resolutions, and renders calibrated, auditable auto-handle vs. human escalation decisions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/codebyArya-bit/hiver-sde-takehome-apple-support-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/codebyArya-bit/hiver-sde-takehome-apple-support-agent/actions)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(15%2F15)-brightgreen.svg)]()
[![Reproduction Time](https://img.shields.io/badge/Reproduction%20Time-%3C10%20Seconds-success.svg)]()
[![Data Leakage](https://img.shields.io/badge/Thread%20Leakage-0%20Overlaps-success.svg)]()
[![Brand](https://img.shields.io/badge/Brand-%40AppleSupport-black.svg)]()

---

## 💡 Quick Example: How the Agent Works

## 💡 Quick Examples: How the Agent Works

To understand the system in 30 seconds, here are two end-to-end execution traces demonstrating its dual mechanisms:

### Example 1: Intent-Independent Safety Escalation (Security Boundary)
When an account compromise or unauthorized charge occurs, the safety layer intercepts critical security keywords (`hacked`, `compromised`, `unauthorized`) through `POLICY_SECURITY_CREDENTIALS` and enforces immediate escalation with official recovery routing, completely independent of model intent confidence.

**Incoming Customer Tweet:**
> *"SOME ASSHOLE HACKED INTO MY ITUNES AND BOUGHT A BUNCH OF EMO MUSIC ON MY ACCOUNT PLEASE HELP ME CANCEL THIS"*

**Agent Pipeline Execution:**
```json
{
  "intent": "OUT_OF_SCOPE_OTHER",
  "intent_confidence": 0.48,
  "escalation_decision": "ESCALATE",
  "escalation_reason": "Account recovery, credential resets, and 2FA authentication cannot be safely resolved over public social channels.",
  "policy_triggered": "POLICY_SECURITY_CREDENTIALS",
  "draft_reply": "Account security is our top priority. Please head to https://iforgot.apple.com to securely verify and recover your Apple ID. If you're still locked out, DM us to assist further.",
  "used_evidence": []
}
```

### Example 2: Grounded Historical RAG (Actionable Resolution Extraction)
For routine technical troubleshooting, the agent retrieves verified historical `@AppleSupport` resolutions, extracts imperative troubleshooting clauses, and incorporates them directly into the response:

**Incoming Customer Tweet:**
> *"This iphone update is crap! My screen brightness won't stay where I put it. Fix this"*

**Agent Pipeline Execution:**
```json
{
  "intent": "IOS_SOFTWARE_UPDATE",
  "intent_confidence": 0.62,
  "escalation_decision": "AUTO_HANDLE",
  "escalation_reason": "Standard on-device display settings and brightness controls addressable with public guidance.",
  "policy_triggered": "NONE_AUTO_HANDLED",
  "draft_reply": "We're happy to help! Tap Settings > Display & Brightness to verify your settings. Check out more troubleshooting steps here: https://support.apple.com/HT204204",
  "used_evidence": [
    {
      "conversation_id": "3e87cfe666859a0b2c248aa4ab5a6f0f",
      "similarity": 0.1539,
      "resolution_snippet": "Tap Settings > Display & Brightness to verify your display preferences."
    }
  ],
  "extracted_action_used": true
}
```

---

## 📊 Headline Benchmark Results

Evaluated across the **200 hand-audited holdout Golden Evaluation Set** (`data/golden_eval_set.json`). The holdout gold set is strictly thread-disjoint from both the 600 training threads and the 1,000-item historical retrieval KB (with training threads contained within the historical KB):

| Metric Dimension | Trivial Baseline (Always Auto-Handle) | Simple Baseline (Naive Bayes + 1-NN) | Proposed AI Agent (RAG + Policy) | Real-World Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Classification Accuracy** | 35.5% | 54.5% | **62.0%** | Out-of-sample generalization across 7 domain intents |
| **Intent Macro F1** | 0.075 | 0.264 | **0.476** | Balanced performance across minority and majority intents |
| **Brier Calibration Score (Lower=Better)** | 1.000 | 0.795 | **0.605** | Superior statistical probability calibration directly from model |
| **Expected Calibration Error (ECE)** | 0.355 | 0.360 | **0.049** | Highly calibrated confidence (predicted confidence tracks empirical accuracy) |
| **Escalation Decision Accuracy** | 74.0% | 75.0% | **71.0%** | Lower raw accuracy due to safety-biased escalation, but much higher recall on cases requiring humans |
| **Escalation Recall (Safety-Critical)** | **0.0%** | **3.9%** | **69.2%** | Catches 36 of 52 escalations; baselines miss 96% to 100% |
| **Escalation Precision** | 0.0% | 100.0% | **46.2%** | Trade-off: 28.4% false escalation rate on auto-handle queries in exchange for 69.2% safety recall |
| **Escalation F2 Score (Recall-Weighted)** | 0.000 | 0.048 | **0.629** | Recall weighted 2x vs. precision, reflecting enterprise safety priority |
| **False Escalation Rate (Lower=Better)**| 0.0% | 0.0% | **28.4%** | Trade-off: accepts ~28% false alarms to protect customer accounts |
| **Illustrative 5:1 Risk Penalty (5*FN + 1*FP)** | 260 | 250 | **122** | Illustrative offline penalty reduced by over 51% |
| **SacreBLEU Score** | 0.2 | 0.4 | **4.1** | Significant lexical alignment over canned baseline macros |
| **Twitter Char Limit Compliance (<280)** | 100.0% | 95.0% | **100.0%** | Zero tweet truncation or broken URL artifacts |
| **Official Apple Domain Inclusion Rate** | 0.0% | 0.0% | **79.5%** | Verified canonical Apple domains (`support.apple.com`, `iforgot.apple.com`) |
| **Intent-Link Relevance Rate** | 0.0% | 0.0% | **75.8%** | Canonical URL matches classified customer issue domain |
| **Heuristic: Groundedness (1–5)** | 4.20 | 4.06 | **4.78** | Factual grounding in verified historical resolution precedents |
| **Heuristic: Brand Voice & Empathy (1–5)** | 5.00 | 4.36 | **4.58** | Professional, empathetic Apple tone within single-tweet limits |
| **Heuristic: Actionability (1–5)** | 4.20 | 4.33 | **4.80** | Concrete step-by-step guidance and canonical navigation paths |
| **Heuristic: Escalation Appropriateness (1–5)**| 4.14 | 4.18 | **4.42** | Safe triage decisions aligned with safety and compliance policies |
| **Heuristic: Overall Quality Score (1–5)** | 4.39 | 4.23 | **4.65** | Holistic quality superiority over both baselines |

*Benchmark execution runtime: **~6.7 seconds on standard CPU** (reproduces offline with zero API keys).*

### Candidate Human Annotator vs. LLM-as-a-Judge Agreement ($N=50$)
The candidate manually scored 50 frozen outputs using the same rubric without viewing the LLM ratings, then compared the two rating sets:
- **Pearson Correlation ($r$)**: **0.920** (Strong linear tracking)
- **Spearman Rank Correlation ($\rho$)**: **0.766** (Consistent ordinal quality ranking)
- **Mean Absolute Error (MAE)**: **0.226 points** on a 1–5 scale (**100.0% within 0.5 points**)
- **Groundedness Agreement**: 100% of ratings within 0.5 points with MAE = 0.360. Restricted score variance (both raters awarding >4.0 due to verified Apple URLs) accounts for lower linear variance ($r=0.156$) while absolute agreement remains high.
- **Cryptographic Hash Verification**: 100% of evaluated pairs match candidate SHA256 input hashes, ensuring both rating files refer to the same frozen outputs and preventing accidental reuse of ratings when model outputs change.

---

## 📁 Key Deliverables & Documentation

- 📄 **[Technical Report (PDF, <= 6 Pages)](REPORT.pdf)**: Print-ready publication PDF strictly adhering to the $\le 6$ page limit (currently 3 pages, enforced programmatically).
- 📝 **[Full Markdown Report](REPORT.md)**: Problem framing, non-goals, baselines, failure modes with authentic `GOLD_xxx` IDs, the mandatory critique *"What is misleading about my headline number?"*, and future work.
- 📋 **[Engineering Decision Log](DECISION_LOG.md)**: 14 non-obvious engineering decisions and trade-offs.
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
               | - Calibrated Logistic Regression (ECE = 0.049)   |
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
*Executes 15 comprehensive unit & regression tests (leakage, single-turn evaluation integrity, cryptographic hash verification, calibration, all 12 adversarial stress cases).*

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
