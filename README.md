# AI Customer Support Agent for @AppleSupport

> **Hiver SDE Intern Take-Home Assignment**: An end-to-end, evaluation-focused AI Support Agent prototype with calibrated safety guardrails built on real Twitter customer support data. Classifies incoming customer messages into domain intents, drafts grounded replies matching historical brand resolutions, and renders calibrated, auditable escalation decisions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(6%2F6)-brightgreen.svg)]()
[![Reproduction Time](https://img.shields.io/badge/Reproduction%20Time-%3C10%20Seconds-success.svg)]()
[![Data Leakage](https://img.shields.io/badge/Thread%20Leakage-0%20Overlaps-success.svg)]()
[![Brand](https://img.shields.io/badge/Brand-%40AppleSupport-black.svg)]()

---

## Headline Benchmark Results

Evaluated across the **200 hand-labelled holdout Golden Evaluation Set** (`data/golden_eval_set.json`) with a **verified zero-leakage conversation-thread split**:

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

*Benchmark execution runtime: **~6.2 seconds on standard CPU** (guaranteeing sub-15 minute reproduction with zero API dependencies).*

---

## Key Deliverables & Documentation

- 📄 **[Technical Report (PDF, <= 6 Pages)](REPORT.pdf)**: Print-ready publication PDF strictly adhering to the $\le 6$ page limit.
- 📝 **[Full Markdown Report](REPORT.md)**: Problem framing, non-goals, baselines, failure modes, the mandatory critique *"What is misleading about my headline number?"*, and future work.
- 📋 **[Engineering Decision Log](DECISION_LOG.md)**: 14 non-obvious engineering decisions and trade-offs.
- 🏷️ **[Golden Set Sampling & Labeling Note](data/SAMPLING_AND_LABELING_NOTE.md)**: Documentation of thread-level partitioning, stratification rules, difficulty tiers, and boundary guidelines.
- 📚 **[Attributions & Citations](ATTRIBUTIONS.md)**: Citations for datasets (Kaggle CC BY-NC-SA 4.0), research papers, open-source libraries, and Apple documentation.

---

## Architecture Overview

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
   | - Tone & Brand Voice conditioning     |   | - Decision: Auto-Handle vs Escalate|
   | - Verified Support Link grounding     |   | - Stated Reason Generation         |
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
                   | - Source Citations & Similar Cases        |
                   +-------------------------------------------+
```

---

## 🚀 Quickstart & Reproduction (< 15 Minutes)

The entire pipeline is self-contained and pre-indexed, requiring **zero API keys** and running entirely offline on CPU.

### 1. Clone & Setup Environment
```bash
git clone <repo-url>
cd "Hive Assignement"

# Install dependencies (runs in < 1 minute)
pip install -r requirements.txt
```

### 2. Run the Headline Benchmark
To run the full evaluation harness comparing all three agents across the 200 Golden Evaluation examples:
```bash
python evaluation/run_evaluation.py
```
*Outputs headline performance tables, human-judge inter-rater reliability statistics, and saves results to `evaluation/benchmark_results.json` in under 10 seconds.*

### 3. Run Automated Unit Tests
To verify all pipeline modules, edge cases, length limit guardrails, and metric calculations:
```bash
python -m pytest tests/test_pipeline.py
```

### 4. Interactive Live Agent Demo
Test the agent interactively with any custom customer tweet:
```bash
python -c "
from src.agent import AppleSupportAgent
agent = AppleSupportAgent().initialize()
result = agent.process_message('My phone battery drops from 80% to 10% in 15 minutes and gets burning hot!')
import json
print(json.dumps(result, indent=2))
"
```

---

## Evaluation Mode Transparency

* **Deterministic Calibrated Multi-Rubric Mode (Default)**: Executes a CPU-based multi-rubric evaluator grading factual grounding, official URL presence, empathy markers, and escalation alignment. Executes in **~6.2 seconds** without API keys.
* **Generative LLM-as-a-Judge Mode (Optional)**: If `OPENAI_API_KEY` or `GEMINI_API_KEY` is exported in the environment, the judge can invoke remote LLMs for natural language reasoning chains.
