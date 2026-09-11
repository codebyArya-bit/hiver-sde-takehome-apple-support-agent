# AI Customer Support Agent for @AppleSupport

> **Hiver SDE Intern Take-Home Assignment**: An end-to-end, evaluation-focused AI Customer Support Agent prototype for `@AppleSupport` built on real Twitter customer support data. Classifies incoming customer messages into domain intents, drafts grounded replies using historical brand resolutions, and renders calibrated, auditable auto-handle vs. human escalation decisions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/codebyArya-bit/hiver-sde-takehome-apple-support-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/codebyArya-bit/hiver-sde-takehome-apple-support-agent/actions)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(15%2F15)-brightgreen.svg)]()
[![Reproduction Time](https://img.shields.io/badge/Reproduction%20Time-%3C10%20Seconds-success.svg)]()
[![Data Leakage](https://img.shields.io/badge/Thread%20Leakage-0%20Overlaps-success.svg)]()
[![Brand](https://img.shields.io/badge/Brand-%40AppleSupport-black.svg)]()

---

## 📊 Headline Benchmark Results

Evaluated across the **200 hand-labelled holdout Golden Evaluation Set** (`data/golden_eval_set.json`) with a **verified zero-leakage conversation-thread split** (zero thread overlap between 600 train threads, 1,000 KB threads, and 200 holdout gold threads):

| Metric Dimension | Trivial Baseline (Always Auto-Handle) | Simple Baseline (Naive Bayes + 1-NN) | Proposed AI Agent (RAG + Policy) | Real-World Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Classification Accuracy** | 35.5% | 54.0% | **61.0%** | Out-of-sample generalization across 7 domain intents |
| **Intent Macro F1** | 0.075 | 0.258 | **0.493** | Balanced performance across minority and majority intents |
| **Brier Calibration Score (Lower=Better)** | 1.000 | 0.812 | **0.615** | Superior statistical probability calibration directly from model |
| **Expected Calibration Error (ECE)** | 0.355 | 0.365 | **0.054** | Highly calibrated confidence (predicted confidence tracks empirical accuracy) |
| **Escalation Decision Accuracy** | 71.0% | 71.0% | **72.5%** | Higher overall triage correctness across safety boundaries |
| **Escalation Recall (Safety-Critical)** | **0.0%** | **1.7%** | **65.5%** | Baselines miss 98.3% to 100% of cases needing humans |
| **Escalation Precision** | 0.0% | 50.0% | **52.0%** | Balanced triage efficiency preventing agent queue overload |
| **Escalation F2 Score (Recall-Weighted)** | 0.000 | 0.021 | **0.623** | Recall weighted 2x vs. precision, reflecting enterprise safety priority |
| **False Escalation Rate (Lower=Better)**| 0.0% | 0.7% | **24.6%** | Trade-off: accepts ~24% false alarms to protect customer accounts |
| **Weighted Risk Penalty (5*FN + 1*FP)** | 290 | 286 | **135** | Total operational risk penalty slashed by over 50% |
| **SacreBLEU Score** | 0.2 | 0.3 | **4.5** | Significant lexical alignment over canned baseline macros |
| **Twitter Char Limit Compliance (<280)** | 100.0% | 95.0% | **100.0%** | Zero tweet truncation or broken URL artifacts |
| **Official Apple Domain Inclusion Rate** | 0.0% | 0.0% | **80.5%** | Verified canonical Apple domains (`support.apple.com`, `iforgot.apple.com`) |
| **Intent-Link Relevance Rate** | 0.0% | 0.0% | **74.8%** | Canonical URL matches classified customer issue domain |
| **Heuristic: Groundedness (1–5)** | 4.20 | 4.06 | **4.79** | Factual grounding in verified historical resolution precedents |
| **Heuristic: Brand Voice & Empathy (1–5)** | 5.00 | 4.36 | **4.58** | Professional, empathetic Apple tone within single-tweet limits |
| **Heuristic: Actionability (1–5)** | 4.20 | 4.33 | **4.80** | Concrete step-by-step guidance and canonical navigation paths |
| **Heuristic: Escalation Appropriateness (1–5)**| 4.02 | 4.03 | **4.40** | Safe triage decisions aligned with safety and compliance policies |
| **Heuristic: Overall Quality Score (1–5)** | 4.36 | 4.20 | **4.65** | Holistic quality superiority over both baselines |

*Benchmark execution runtime: **~6-9 seconds on standard CPU** (reproduces offline with zero API keys).*

### Ground Truth Human vs. LLM-as-a-Judge Agreement ($N=50$)
To ensure the automated evaluator tracks human judgment rather than surface heuristics, an authentic paired blind study was conducted on identical frozen model outputs with cryptographic SHA256 input hash assertions:
- **Pearson Correlation ($r$)**: **0.893** (Strong linear tracking)
- **Spearman Rank Correlation ($\rho$)**: **0.810** (Consistent ordinal quality ranking)
- **Mean Absolute Error (MAE)**: **0.191 points** on a 1–5 scale (**100.0% within 0.5 points**)
- **Cryptographic Hash Verification**: 100% of evaluated pairs match candidate SHA256 input hashes, guaranteeing zero stale score reuse.

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
               | - Calibrated Logistic Regression (ECE = 0.054)   |
               +--------------------------------------------------+
                                         |
                        +----------------+----------------+
                        |                                 |
                        v                                 v
   +---------------------------------------+   +------------------------------------+
   | 2. Grounded Resolution Engine (RAG)  |   | 3. Escalation Decision Engine      |
   | - Sparse TF-IDF retrieval over        |   | - Risk & Compliance Safety Policy  |
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

### 2. Verify Data Integrity & Zero Data Leakage
```bash
python scripts/verify_splits.py
```
*Confirms 0 thread overlaps between Train (600), KB (1000), and Gold (200) splits and validates turn separation.*

### 3. Run the Headline Benchmark Harness
```bash
python evaluation/run_evaluation.py
```
*Executes all three agents on the 200 Golden Evaluation examples, generates comparison tables and human-judge agreement stats, and outputs `evaluation/benchmark_results.json` and `evaluation/error_analysis.json` in ~6-9 seconds.*

### 4. Run Automated Test Suite
```bash
python -m pytest tests/test_pipeline.py -v
```
*Executes 15 comprehensive unit & regression tests (leakage, turn separation, cryptographic hash verification, calibration, adversarial stress tests).*

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
