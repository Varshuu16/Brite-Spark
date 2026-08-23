# Architectural Decisions Record — BriteSpark Problem 1

## Day 2 Surprise Challenge: Amendment No. 2026-01 & Temporal Grounding

---

### 1. Retention of `data/policy-manual.md` as an Unmodified Base Corpus

#### Context:
Amendment No. 2026-01 was issued on 12 February 2026, taking effect on 1 March 2026. The consolidated text of the manual represents policy as at 31 December 2025.

#### Decision:
We retain `data/policy-manual.md` 100% byte-for-byte unchanged. The amendment is loaded as an independent, modular layer from `data/Amendment No. 2026-01.md`.

#### Rationale:
- In real-world statutory and administrative systems, past legal text cannot be overwritten because claims, determinations, and appeals that arose before an amendment took effect must be judged against the policy text in force on the relevant historical date.
- Preserving the original document ensures total backward compatibility, strict auditability, and historical correctness.

---

### 2. Multi-Version Clause Architecture & Modular Delta Layers

#### Context:
Clauses such as §6.4.1(a) (earnings disregard) and §4.3.2 (reporting deadline) have two valid states depending on the claim/determination date.

#### Decision:
We extended `PolicyClause` with temporal and provenance metadata (`source_document`, `effective_date`, `amended_by`, `amends_clause_id`, `transitional_rule`, `is_amendment`, `is_transitional`). When an amendment is loaded, it creates discrete amended clause objects, inserted clauses (such as §10.5.3A), and transitional provisions (§5.1, §5.2, §5.3) alongside the historical clauses.

#### Rationale:
- Allows the retriever and answer generator to access both versions simultaneously when a query has no date or spans a transition.
- Avoids destructively mutating the base clause objects.

---

### 3. Deterministic Temporal Classification Without LLM Dependency

#### Context:
User queries may refer to specific dates (e.g. "February 2026", "15 March 2026"), date ranges ("spanning February to April 2026"), or no date at all.

#### Decision:
We implemented a deterministic temporal parser (`src/temporal.py`) using robust regex date extraction and legal event classification (`QueryEventType.DETERMINATION`, `QueryEventType.CHANGE_OF_CIRCUMSTANCES`, `QueryEventType.SPANNING_PERIOD`, `QueryEventType.GENERAL`).

#### Rationale:
- LLMs are prone to inconsistent date parsing and arithmetic errors.
- A deterministic classification guarantees reproducible classification into `PRE_AMENDMENT`, `POST_AMENDMENT`, `SPANNING`, or `UNSPECIFIED` before retrieval and prompt generation occur.

---

### 4. Handling of Transitional Provisions (§5.1 vs §5.2 vs §5.3)

#### Policy Rules:
1. **§5.1 (Determination-Date Rule):**
   - Applies to earnings disregard (§6.4.1(a)), income thresholds (§6.6.1), and sanctions (§10.5.2, §10.5.3A).
   - Governed strictly by the date the *determination* is made (on or after 1 March 2026), even if the underlying claim period was earlier.
2. **§5.2 (Event-Date Rule):**
   - Applies to recipient change reporting (§4.3.2) and overpayment safe harbor (§9.1.4).
   - Governed strictly by the date the *change of circumstances occurred*. Changes before 1 March 2026 adhere to the 10-day period (§4.3.2) / 30-day safe harbor (§9.1.4); changes on or after 1 March 2026 adhere to the 14-day period.
3. **§5.3 (Spanning Period Rule):**
   - Claims spanning 1 March 2026 use daily rates in force on each day, apportioned under §7.4.3.

#### Implementation:
The temporal engine identifies the event type from the user query and pairs the substantive clauses with the exact transitional provision.

---

### 5. Grounded Answering for Unspecified Dates

#### Decision:
When a user does not specify a date (e.g. "What is the earnings disregard?"), the system does not guess today's date. Instead, it retrieves both the historical clause and the amended clause along with transitional rules, and prompts Gemini to explicitly describe:
1. The prior rule before 1 March 2026 ($120 / 10 days).
2. The current amended rule taking effect 1 March 2026 ($175 / 14 days).
3. The transitional conditions under which each applies.

---

### 6. Retrospective Design Reflection

*What would we have done differently if temporal amendments had been known from Day 1?*

1. **Bi-Temporal Clause Indexing:**
   - We would have designed `PolicyClause` from the start with `valid_from` and `valid_to` timestamps and built an interval-tree index in `PolicyRetriever` for $O(1)$ point-in-time point queries.
2. **Graph-Based Delta Patching:**
   - Rather than loading amendments as parallel clause lists, we would have implemented a formal Policy Graph where amendments attach as edge revisions with condition predicates (`determination_date >= 2026-03-01`).

---

### 7. Deterministic Substantive Policy Conflict Handling (Part 6)

#### Context:
Policy corpora often contain contradictory provisions across different administrative chapters (e.g. §4.3.2 establishing a 10-day reporting deadline vs §9.1.4 referencing 30 calendar days for overpayment notifications, or general sanction rules vs statutory exceptions). The system must never silently choose one rule or hallucinate precedence without evidence.

#### Decision:
We introduced a dedicated, deterministic conflict detection module (`src/conflict.py`) and a structured `PolicyConflict` data model.

#### Architectural Principles:
1. **Separation of Temporal Evolution from Substantive Conflict:**
   - Pre-amendment vs post-amendment version differences (e.g. §4.3.2 original 10-day vs amended 14-day rule) are resolved by the temporal engine (`§5.1`, `§5.2`, `§5.3`). These are tracked as chronological temporal versions, NOT unresolved conflicts.
2. **Deterministic Contradiction Detection:**
   - Evaluates retrieved clauses for direct numeric discrepancies and sanction conflicts (e.g. sanction imposition vs prohibition under §10.5.3A).
3. **Transparent Answer Generation & Citation Validation:**
   - When a conflict is detected, the prompt instructs Gemini to explicitly describe the discrepancy, cite both conflicting clauses, and state that the provided evidence does not establish precedence.
   - All cited conflicting clauses must be present in retrieved evidence, maintaining full grounding (`grounded = True`, `conflicts_detected = True`).

---

### 8. Deterministic Multi-Dimensional Evaluation Framework (Part 7)

#### Context:
Evaluating an administrative policy AI requires rigorous verification across retrieval accuracy, temporal reasoning, citation traceability, refusal safety, and conflict detection. Relying solely on live generative LLM calls for evaluation introduces non-determinism, network flakiness, latency, and uncontrolled prompt drift.

#### Decision:
We established a comprehensive deterministic evaluation framework (`tests/evaluation_dataset.py`, `tests/run_evaluation.py`, and `tests/test_evaluation.py`) with 21 structured test cases spanning 15 distinct categories.

#### Core Evaluation Principles:
1. **Zero-Flake Deterministic Evaluation:**
   - Core regression and evaluation scoring runs completely offline using controlled mock answer generators and deterministic validators. This guarantees 100% reproducible results without API quotas, rate limits, or network variability.
2. **Multi-Dimensional Quality Metrics:**
   - The framework computes 6 distinct deterministic metrics:
     - **Evidence Retrieval Accuracy:** Expected clause hit rate in top-$k$ results.
     - **Temporal Classification Accuracy:** Exact classification of legal temporal status (`PRE_AMENDMENT`, `POST_AMENDMENT`, `SPANNING`, `UNSPECIFIED`).
     - **Citation Validity Rate:** Strict verification that all cited clauses exist in retrieved evidence.
     - **Citation Completeness Rate:** Verification that substantive policy assertions are supported by citations.
     - **Refusal Safety Accuracy:** Immediate structured refusal for out-of-domain and zero-evidence queries.
     - **Conflict Detection Accuracy:** Correct identification of substantive contradictions versus temporal versioning.
3. **Rejection of Artificial "ML Accuracy" for Generated Prose:**
   - Rather than computing noisy similarity metrics (e.g. ROUGE, BLEU, or arbitrary LLM-as-judge scores) over natural-language prose, our framework evaluates exact factual constraints: presence of required policy numbers/dates, exact citation strings, absence of hallucinations, and boolean grounding flags.
