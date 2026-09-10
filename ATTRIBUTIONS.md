# Attributions and Citations of Borrowed Assets

This document transparently lists all datasets, external software libraries, domain link structures, evaluation rubrics, and conceptual frameworks utilized or adapted in this project, in strict compliance with the assignment rules.

---

## 1. Primary Datasets
* **Customer Support on Twitter (Kaggle)**
  * **Original Authors**: `thoughtvector`
  * **Source URL**: [https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
  * **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
  * **Usage**: Filtered exclusively for company handle `@AppleSupport`. Partitioned strictly into 1,000 historical knowledge base interaction pairs (`apple_support_kb.json`), a 600-thread disjoint training split (`apple_train_set.json`), and 200 holdout candidates for the golden evaluation set (`golden_eval_set.json`).
* **Customer Support on Twitter Conversation Threads (Hugging Face)**
  * **Curator / Uploader**: `TNE-AI` (`TNE-AI/customer-support-on-twitter-conversation`)
  * **Source URL**: [https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation](https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation)
  * **Usage**: Used to reconstruct multi-turn customer inquiry to support response threads with intact conversation IDs.

---

## 2. Research Methodologies & Theoretical Frameworks
* **LLM-as-a-Judge Evaluation Framework**:
  * **Citation**: Lianmin Zheng, Wei-Lin Chiang, Hao Zhang, et al. *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*, Advances in Neural Information Processing Systems (NeurIPS 2023), Track on Datasets and Benchmarks.
  * **Adaptation**: Adapted multi-criteria pairwise and point-wise rubric grading to customer support operational standards across 4 axes: Groundedness, Brand Voice, Actionability, and Escalation Appropriateness.
* **Inter-Rater Agreement & Statistical Reliability**:
  * **Cohen's Kappa ($\kappa$)**: Cohen, Jacob. *"A Coefficient of Agreement for Nominal Scales"*, Educational and Psychological Measurement, 20(1), 37–46 (1960). Used to measure binned categorical agreement between human annotations and automated judge ratings beyond chance.
  * **Inter-Rater Reliability Thresholds**: Landis, J. Richard, & Gary G. Koch. *"The measurement of observer agreement for categorical data"*, Biometrics, 159–174 (1977). Benchmark used to interpret $\kappa > 0.80$ as "almost perfect agreement".
  * **Linear & Rank Correlation**: Pearson Correlation ($r$) and Spearman Rank Correlation ($\rho$) computed via `scipy.stats`.

---

## 3. Official Brand Knowledge Base & Domain Links
* **Apple Official Support Documentation**:
  * Battery & Power Repair: `https://support.apple.com/iphone/repair/battery-power`
  * iPhone Battery Health & Performance: `https://support.apple.com/HT208387`
  * iOS Software Update Troubleshooting: `https://support.apple.com/HT204204`
  * Apple ID Password & Account Recovery: `https://iforgot.apple.com`
  * App Store & iTunes Billing / Refund Portal: `https://reportaproblem.apple.com`
  * iPhone User Guide: `https://support.apple.com/guide/iphone`
  * Apple Customer Feedback: `https://www.apple.com/feedback`
  * Direct Message Escalation Portal: `https://twitter.com/messages/compose`

---

## 4. Software Libraries & Open Source Tools
* **Core Machine Learning**: `scikit-learn` (v1.7.2), `numpy` (v2.3.3), `scipy` (v1.16.2), `joblib` (v1.5.2).
* **Deep Learning & NLP**: `torch` (v2.8.0), `transformers` (v4.57.0), `datasets` (v4.1.1), `sacrebleu` (v2.5.1).
* **CLI & Rendering**: `rich` (v14.1.0), `tabulate` (v0.9.0).
* **Testing**: `pytest` (v8.4.2).
* **Document Generation**: `reportlab` (v4.4.5).

---

## 5. AI Coding Assistance Disclosure
* **Tooling Used**: Google DeepMind Antigravity AI pair programming assistant.
* **Scope of Assistance**: Scaffolded Python boilerplate, extracted dataset slices, assisted in calculating matrix metrics, and authored LaTeX/ReportLab styling scripts.
* **Integrity Commitment**: All code, architecture designs, mathematical metrics, and failure analyses have been audited, modified, and understood by the candidate, ready for live walkthrough and code-level modification during technical interview rounds.
