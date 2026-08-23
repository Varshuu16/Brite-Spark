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
