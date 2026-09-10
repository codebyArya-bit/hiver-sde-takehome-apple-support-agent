# AI Customer Support Agent for @AppleSupport

> **Hiver SDE Intern Take-Home Assignment**: An end-to-end, evaluation-focused AI Customer Support Agent prototype for `@AppleSupport` built on real Twitter customer support data. Classifies incoming customer messages into domain intents, drafts grounded replies using historical brand resolutions, and renders calibrated, auditable auto-handle vs. human escalation decisions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(11%2F11)-brightgreen.svg)]()
[![Reproduction Time](https://img.shields.io/badge/Reproduction%20Time-%3C10%20Seconds-success.svg)]()
[![Data Leakage](https://img.shields.io/badge/Thread%20Leakage-0%20Overlaps-success.svg)]()
[![Brand](https://img.shields.io/badge/Brand-%40AppleSupport-black.svg)]()

---

## 📊 Headline Benchmark Results

Evaluated across the **200 hand-labelled holdout Golden Evaluation Set** (`data/golden_eval_set.json`) with a **verified zero-leakage conversation-thread split** (zero thread overlap between 600 train threads, 1,000 KB threads, and 200 holdout gold threads):

| Metric Dimension | Trivial Baseline (Always Auto-Handle) | Simple Baseline (Naive Bayes + 1-NN) | Proposed AI Agent (RAG + Policy) | Real-World Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Classification Accuracy** | 35.5% | 54.5% | **64.5%** | Out-of-sample generalization across 7 domain intents |
| **Intent Macro F1** | 0.075 | 0.262 | **0.535** | Superior balance on under-represented intents |
| **Escalation Decision Accuracy** | 71.0% | 71.0% | **76.5%** | Higher overall triage accuracy |
| **Escalation Recall (Safety-Critical)** | **0.0%** | **1.7%** | **70.7%** | Baselines miss 98.3% to 100% of cases needing humans |
| **Escalation Precision** | 0.0% | 50.0% | **57.8%** | Balanced triage efficiency preventing agent overload |
| **False Escalation Rate (Lower=Better)**| 0.0% | 0.7% | **21.1%** | Trade-off: accepts ~21% false alarms to catch critical risks |
| **SacreBLEU Score** | 0.2 | 0.3 | **4.7** | Significant improvement over generic canned baselines |
| **Twitter Char Limit Compliance (<280)** | 100.0% | 92.5% | **100.0%** | Zero tweet truncation or broken URL artifacts |
| **Official Domain Link Validity** | 0.0% | 0.0% | **79.5%** | Provides verified official Apple URLs vs stale redirects |
| **Intent-Link Relevance Rate** | 0.0% | 0.0% | **73.0%** | Canonical URL matches classified problem domain |
| **Heuristic: Groundedness (1–5)** | 4.20 | 4.05 | **4.78** | Factual grounding in historical troubleshooting steps |
| **Heuristic: Brand Voice & Empathy (1–5)** | 5.00 | 4.30 | **4.61** | Professional Apple tone without sounding robotic |
| **Heuristic: Actionability (1–5)** | 4.20 | 4.33 | **4.81** | High practical utility and clear next actions |
| **Heuristic: Escalation Appropriateness (1–5)**| 4.00 | 4.01 | **4.46** | Safe triage decisions aligned with enterprise risk policy |
| **Heuristic: Overall Quality Score (1–5)** | 4.35 | 4.17 | **4.67** | Clear superiority across combined holistic criteria |

*Benchmark execution runtime: **~7.8 seconds on standard CPU** (reproduces offline with zero API keys).*

### Ground Truth Human vs. LLM-as-a-Judge Agreement ($N=50$)
To ensure the automated evaluator tracks human judgment rather than surface heuristics, a paired blind study was conducted on identical frozen model outputs:
- **Pearson Correlation ($r$)**: **0.964** (Near-perfect linear tracking)
- **Spearman Rank Correlation ($\rho$)**: **0.986** (Consistent ordinal quality ranking)
- **Mean Absolute Error (MAE)**: **0.108 points** on a 1–5 scale (100% within 0.5 points)
- **Cohen's Kappa ($\kappa$)**: **0.733** (Substantial inter-rater agreement beyond chance)

---

## 📁 Key Deliverables & Documentation

- 📄 **[Technical Report (PDF, <= 6 Pages)](REPORT.pdf)**: Print-ready publication PDF strictly adhering to the $\le 6$ page limit (currently 3 pages, enforced programmatically).
- 📝 **[Full Markdown Report](REPORT.md)**: Problem framing, non-goals, baselines, failure modes with authentic `GOLD_xxx` IDs, the mandatory critique *"What is misleading about my headline number?"*, and future work.
- 📋 **[Engineering Decision Log](DECISION_LOG.md)**: 14 non-obvious engineering decisions and trade-offs.
- 🏷️ **[Golden Set Sampling & Labeling Note](data/SAMPLING_AND_LABELING_NOTE.md)**: Documentation of thread-level partitioning, turn separation, stratification rules, difficulty tiers, and boundary guidelines.
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
               +--------------------------------------------------+
                                         |
                        +----------------+----------------+
                        |                                 |
                        v                                 v
   +---------------------------------------+   +------------------------------------+
   | 2. Grounded Resolution Engine (RAG)  |   | 3. Escalation Decision Engine      |
   | - Semantic vector index over          |   | - Risk & Capability Policy         |
   |   1,000 historical resolutions        |   | - Confidence & Ambiguity Scorer    |
   | - Concrete resolution extraction      |   | - Decision: Auto-Handle vs Escalate|
   | - Official Apple URL whitelist        |   | - Stated Reason Generation         |
   | - Twitter <280-char compliance        |   |                                    |
   +---------------------------------------+   +------------------------------------+
                        \                                 /
                         \                               /
                          v                             v
                   +-------------------------------------------+
                   | Final Agent Output:                       |
                   | - Intent + Confidence                     |
                   | - Grounded Draft Reply (<280 chars)       |
                   | - Routing Decision (Auto/Escalate)        |
                   | - Stated Escalation Reason                |
                   | - Structured Evidence Citations           |
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

# Install minimal, pinned dependencies (< 30 seconds)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Verify Data Integrity & Zero Data Leakage
```bash
python scripts/prepare_dataset.py --verify-only
```
*Confirms 0 thread overlaps between Train (600), KB (1000), and Gold (200) splits.*

### 3. Run the Headline Benchmark Harness
```bash
python evaluation/run_evaluation.py
```
*Executes all three agents on the 200 Golden Evaluation examples, generates comparison tables and human-judge agreement stats, and outputs `evaluation/benchmark_results.json` and `evaluation/error_analysis.json` in ~7.8 seconds.*

### 4. Run Automated Tests
```bash
python -m pytest tests/test_pipeline.py -q
```

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
