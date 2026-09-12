# Engineering & Design Decision Log

A chronological record of 14 non-obvious technical and architectural decisions made while designing, implementing, and evaluating the `@AppleSupport` AI Support Agent.

---

### 1. Brand Selection: Choosing `@AppleSupport` Over `@AmazonHelp`
* **Decision**: Focus exclusively on `@AppleSupport` rather than retail giants like `@AmazonHelp` or telecom brands (`@TMobileHelp`).
* **Rationale**: Apple Support on Twitter exhibits a distinct, highly structured bifurcation between deterministic self-service troubleshooting (settings, reboot, orientation lock, battery health checks) and hard escalation boundaries (security lockout, hardware damage, financial billing). This sharp boundary makes evaluation of *trust* and *escalation safety* much more rigorous and verifiable.
* **Trade-off**: Requires deeper modeling of device-specific technical terminology (e.g. iOS versions, DFU mode, battery health percentage) compared to generic e-commerce tracking inquiries.

---

### 2. Taxonomy Granularity: 7 Intents Instead of Fine-Grained (77) or Coarse (3)
* **Decision**: Bound the intent space to 7 domain-grounded classes: `BATTERY_AND_HARDWARE`, `IOS_SOFTWARE_UPDATE`, `APPLE_ID_AND_ICLOUD`, `APP_STORE_AND_BILLING`, `DEVICE_SETUP_AND_USAGE`, `CUSTOMER_FEEDBACK_COMPLAINT`, and `OUT_OF_SCOPE_OTHER`.
* **Rationale**: Coarse 3-class taxonomies (Hardware, Software, Account) hide safety-critical distinctions (e.g. a billing refund vs. a software glitch). Conversely, a 77-class taxonomy (like Banking77) causes severe data sparsity over Twitter's 140/280-character noisy inputs, reducing classifier calibration. 7 intents capture 96%+ of enterprise operational triage categories without label fragmentation.
* **Trade-off**: Compound tweets (e.g. "I updated to iOS 11 and now my battery drains and Apple ID is locked") must be assigned to the dominant safety-critical intent rather than multi-labeled.

---

### 3. Asymmetric Metric Optimization: Prioritizing Safety Recall Over Precision in Escalation
* **Decision**: Prioritize **Escalation Recall** (catching issues needing human agents) as a tier-1 critical metric, accepting a slight false escalation penalty over any risk of false auto-handling. Formalized via **Escalation F2 Score** and a **5:1 Weighted Risk-Cost Penalty** (5 points for a missed safety escalation vs. 1 point for an unnecessary escalation).
* **Rationale**: In customer support operations, an unnecessary human escalation consumes frontline agent capacity and extends response backlogs; conversely, an automated hallucination or failure to escalate on an account takeover, damaged battery, or billing dispute creates severe customer harm, security exposure, and brand trust erosion. Under-escalation is safety-catastrophic; over-escalation is an operational capacity trade-off.
* **Trade-off**: The agent routes ambiguous and safety-boundary queries to human specialists rather than aggressively attempting to auto-resolve everything, resulting in a false escalation rate of ~28.4% while achieving a 69.2% escalation recall and cutting the illustrative risk penalty from 260 to 122.

---

### 4. Zero-Dependency Offline Execution Architecture
* **Decision**: Implement a self-contained local pipeline using calibrated statistical classifiers and sparse TF-IDF retrieval on CPU, rather than relying strictly on paid external LLM APIs (OpenAI / Anthropic / Gemini).
* **Rationale**: Fulfills the strict requirement that any reviewer or recruiter can clone the repository and reproduce headline results in under 15 minutes without configuring API keys, credit cards, or external cloud quotas.
* **Trade-off**: Reply phrasing relies on retrieved verified historical precedents and modular templates rather than unconstrained generative autoregression.

---

### 5. Separate KB Indexing from Evaluation Candidates (Anti-Contamination Partition)
* **Decision**: Partition the scraped conversations such that the 200 holdout golden evaluation examples have zero thread overlap with both the 600 training threads and the 1,000 historical KB entries (the 600 classifier training threads are included within the 1,000 KB entries).
* **Rationale**: Evaluating RAG systems on data present in the vector index produces artificially inflated ROUGE/BLEU scores and circular evaluation. Keeping the golden set completely out of the retrieval index tests true out-of-sample generalization.
* **Trade-off**: Gold set isolation ensures zero data contamination during holdout evaluation.

---

### 6. Dynamic Calibration of Retrieval Similarity Threshold
* **Decision**: Set the default `min_retrieval_similarity` threshold to `0.05` in `EscalationEngine` after empirical percentile analysis over sparse TF-IDF vectors.
* **Rationale**: Due to the extreme brevity of customer tweets (10–20 tokens) and high vocabulary sparsity, cosine similarities on sparse historical resolution text rarely exceed 0.25 even for relevant matches. An aggressive 0.20 threshold caused an >80% false escalation rate. Setting the threshold to 0.05 and pairing it with calibrated policy triggers preserves high recall on safety-critical interactions while preventing unnecessary agent queue flooding.
* **Trade-off**: Relies on domain policy regex triggers rather than vector similarity alone to catch out-of-distribution adversarial prompts.


---

### 7. Explicit Policy-Driven Hard Escalation Layer
* **Decision**: Decouple the escalation engine from pure model confidence, placing a deterministic policy rulebook ahead of model probability.
* **Rationale**: Deep learning models are known to be overconfident on out-of-distribution or adversarial inputs. Handlers for keywords like "lawyer", "sue", "swollen battery", and "stolen credit card" should never be subject to probabilistic model sampling.
* **Trade-off**: Requires maintaining a domain policy ruleset alongside machine learning models.

---

### 8. Hard 280-Character Boundary Enforcement in the Generator
* **Decision**: Implement strict character-budgeting and smart truncation that preserves official Apple Support URLs at the end of every reply.
* **Rationale**: Twitter/X enforces a strict character limit. A reply truncated mid-URL produces a broken 404 link, rendering the entire automated assistance useless.
* **Trade-off**: Extended explanations must be condensed into telegraphic, high-density instructions.

---

### 9. Dual Baseline Design: Trivial (Majority Class) vs. Simple (1-NN + Naive Bayes)
* **Decision**: Contrast the proposed agent against both a trivial majority-rule agent and a classical statistical ML baseline (Naive Bayes + 1-NN Lexical Retrieval).
* **Rationale**: Benchmarking against only a strawman makes performance gains look trivial. Benchmarking against a standard ML baseline proves where domain grounding, calibration, and policy safety specifically add value over traditional text retrieval.
* **Trade-off**: Requires training, evaluating, and maintaining three separate agent implementations in the test harness.

---

### 10. Multi-Dimensional Quality Rubric Over Single Likert Scale
* **Decision**: Score responses on 4 independent axes (Groundedness, Brand Voice, Actionability, Escalation Appropriateness) rather than a single 1–5 "Overall Quality" metric.
* **Rationale**: A reply can have pristine brand voice and empathy while being factually wrong or inappropriately auto-handled. Multi-axis scoring pinpoints the exact failure modes.
* **Trade-off**: Increases computation and evaluation matrix size by 4x.

---

### 11. Inclusion of Non-English and Multilingual Real Queries in Evaluation
* **Decision**: Explicitly retain non-English tweets (Spanish, French) present in the Kaggle dataset in the Golden Evaluation Set.
* **Rationale**: Real social support queues receive multilingual traffic. The system must recognize that it is not equipped for Spanish technical advice and route to native-language queues rather than replying with English macros.
* **Trade-off**: Slightly lowers raw single-language auto-handle throughput.

---

### 12. Verification of URL Integrity & Official Domain Whitelisting
* **Decision**: Enforce standard fallback mappings to whitelisted official Apple domains (`support.apple.com`, `reportaproblem.apple.com`, `iforgot.apple.com`) if historical retrieved links are stale Twitter t.co redirects.
* **Rationale**: Historical Kaggle tweets contain shortened `t.co` URLs that may be expired or ungrounded. Injecting verified, canonical Apple knowledge base anchors guarantees functional customer links.
* **Trade-off**: Requires maintaining a canonical domain link map for each intent.

---

### 13. Stratified Difficulty Sampling for Golden Set (Easy / Medium / Hard)
* **Decision**: Annotate every golden item with a difficulty tier (`EASY`: 27.0%, `MEDIUM`: 55.5%, `HARD`: 17.5%).
* **Rationale**: Prevents benchmark inflation caused by packing the test set with trivial FAQs (e.g. "how do I take a screenshot").
* **Trade-off**: Requires detailed manual review of edge cases (e.g. bulging batteries, minor in-app purchase disputes).


---

### 14. Measuring Human-Judge Agreement via Inter-Rater Reliability (Pearson, Spearman, Kappa, MAE)
* **Decision**: Include authentic mathematical inter-rater agreement statistics in the automated evaluation harness with cryptographic SHA256 input hash assertions.
* **Rationale**: An automated judge cannot be trusted unless its scoring distribution demonstrably aligns with human grading. By hashing the exact evaluation payload (`item_id`, `query`, `gold_intent`, `gold_escalation`, `candidate_reply`, `candidate_escalation`, `rubric_version`), the evaluation harness ensures that both human ratings and LLM ratings refer to the exact same frozen candidate outputs, preventing accidental reuse of ratings when model responses change.
* **Trade-off**: Requires maintaining paired human and LLM evaluation sets with strict hash verification that fails loudly on any candidate text discrepancy.

---

### 15. Single-Source-of-Truth Provenance, Statistical Rigor & Disjoint Split Framing
* **Decision**: Synchronize canonical evaluation metadata, treat undefined Cohen's Kappa strictly as `null` in JSON and `"N/A"` in presentation (never substituting artificial 1.0), lock judge provenance metadata to `gemini-2.5-flash`, and formally frame dataset splits: the 200-item gold holdout is thread-disjoint from both the 600 training examples and the 1,000-item historical retrieval KB (with training threads contained within the KB).
* **Rationale**: Defensibility and honesty in engineering evaluation outweigh inflated or misleading metric representations. Statistical edge cases (e.g. constant rater agreement) must reflect true mathematical properties rather than swallow exceptions.
* **Trade-off**: Requires strict automated synchronization between evaluation artifacts, code provenance schemas, and documentation.

