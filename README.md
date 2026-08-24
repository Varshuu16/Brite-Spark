# BriteSpark Policy Assistant — Problem 1

An evidence-grounded, temporal-aware, conflict-safe policy reasoning engine with deterministic BM25 retrieval, citation verification, and a conversational web interface.

---

## 1. Problem Statement

Administrative policy manuals govern public benefit determinations, award calculations, reporting deadlines, and sanctions. Deploying Large Language Models directly against raw administrative policies introduces critical failure modes:
- **Hallucinated citations & rules:** Models invent provisions or cite non-existent sections.
- **Temporal confusion:** Amendments alter rules on specific dates; models struggle with transitional clauses (§5.1, §5.2, §5.3).
- **Silent contradiction resolution:** When policies contain conflicting provisions across chapters, models silently pick one without alerting the user.
- **Unsafe answers to unindexed topics:** Models answer out-of-domain questions using generic world knowledge rather than refusing based on evidence.

The **BriteSpark Policy Assistant** solves these challenges through a deterministic pipeline combining strict evidence parsing, BM25 retrieval with cross-reference expansion, explicit temporal reasoning, citation extraction and validation, deterministic conflict detection, and a conversational web UI.

---

## 2. Solution Overview

```
                     User Question
                           │
                           ▼
                    Policy Parsing
                           │
                           ▼
           BM25 Retrieval + Cross References
                           │
                           ▼
                   Temporal Reasoning
                           │
                           ▼
                   Conflict Detection
                           │
                           ▼
                 Grounded Gemini Answer
                           │
                           ▼
                  Citation Validation
                           │
                           ▼
              Refusal / Completeness Checks
                           │
                           ▼
                Conversational Web UI
```

---

## 3. Key Features

### 3.1 Temporal Engine & Surprise Challenge
The Surprise Challenge introduced date-sensitive versioning via **Amendment No. 2026-01** (effective 1 March 2026). The temporal engine resolves date-dependent rules by distinguishing semantic date roles:
- **Change Date (§5.2):** Reporting deadlines (14 calendar days vs 10 calendar days) are strictly governed by the date the change occurred.
- **Determination Date (§5.1):** Earnings disregards ($175 vs $120 per month) and calculation adjustments apply to determinations made on/after 1 March 2026, even for earlier claim periods.
- **Spanning Periods (§5.3):** Claim periods spanning across 1 March 2026 trigger daily rate apportionment under §7.4.3.
- **Classification:** Queries are classified into `PRE_AMENDMENT`, `POST_AMENDMENT`, `SPANNING`, or `UNSPECIFIED`. Normal amendment version changes are handled through temporal versioning rather than treated as unresolved conflicts.

### 3.2 Citation Extraction & Validation
- Generated citations (e.g. `§4.3.2`, `Amendment 2026-01 §5.2`, `§10.5.3A`) are extracted and verified against the evidence actually retrieved for that query.
- Unsupported or unretrieved citations are rejected and marked ungrounded (`grounded = False`).
- Plain numbers, dates, currency values (`$175`), and day counts (`10 calendar days`) are filtered out and not treated as citations.

### 3.3 Refusal & Insufficient Evidence Handling
- Fast refusal path before invoking Gemini when retrieved evidence is absent or out-of-domain (e.g. *"What is the capital of France?"*).
- Refusal contract guarantees: `grounded = False`, `insufficient_evidence = True`, `citations = []`.

### 3.4 Substantive Conflict Detection
- Detects substantive contradictions across retrieved evidence (e.g. the 10/14-day recipient reporting rule under §4.3.2 vs the 30-day overpayment notice in §9.1.4).
- Distinguishes chronological amendment updates (resolved temporally) from true substantive contradictions.
- Never silently picks a single rule; highlights the contradiction and cites both provisions.

---

## 4. Web Interface

Built with Flask, HTML, and CSS:
- **Centered Home State:** Clean white background with a centered heading, subtitle, rounded input box, and quick-fill example pills.
- **Conversational Layout:** Transitions into a conversation stream upon submitting a question; previous turns remain visible during the active session.
- **Structured Answers:** Displays grounded response, status metadata (`Grounding`, `Evidence`, `Temporal`, `Conflict`), and citation cards with verbatim policy quotes.
- **Alerts & Reset:** Clear warning banners for detected conflicts and refusals; `+ New Question` button resets to the home state.

---

## 5. Installation

### Prerequisites
- Python 3.10+
- `pip`

```bash
# Clone the repository
git clone https://github.com/Varshuu16/Brite-Spark.git
cd Brite-Spark

# Install dependencies
python -m pip install -r requirements.txt
```

### Environment Configuration (Optional for live Gemini API)
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Linux / macOS
export GEMINI_API_KEY="your-api-key-here"
```
*Note: Without `GEMINI_API_KEY`, the application and test suites run in deterministic offline mode.*

---

## 6. Running the Application

### 6.1 Start the Web UI
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

### 6.2 CLI Demonstration
```bash
python src/answer.py
```

### 6.3 Run Full Regression Test Suite (112 Tests)
```bash
python -m unittest discover tests -v
```
- **112 total tests**, **112 passed** (0 failures, 0 errors, 2 skipped live-network tests when `GEMINI_API_KEY` is not set).

### 6.4 Run Deterministic Evaluation (21 Cases)
```bash
python tests/run_evaluation.py
```

---

## 7. Evaluation Results

Evaluated across 21 structured test cases in `tests/evaluation_dataset.py`:

| Metric Dimension | Score | Description |
| :--- | :--- | :--- |
| **Evidence Retrieval Accuracy** | **100.0%** | Expected governing clauses retrieved in top-$k$ results. |
| **Temporal Classification Accuracy** | **100.0%** | Exact categorization into `PRE_AMENDMENT`, `POST_AMENDMENT`, `SPANNING`, or `UNSPECIFIED`. |
| **Citation Validity Rate** | **95.2%** | Validated citations match retrieved evidence; unretrieved citations are flagged. |
| **Citation Completeness Rate** | **95.2%** | Substantive assertions require citations; un-cited assertions fail. |
| **Refusal Safety Accuracy** | **100.0%** | Immediate structured refusal for out-of-domain and zero-evidence queries. |
| **Conflict Detection Accuracy** | **100.0%** | Substantive contradictions are flagged while temporal versioning is resolved. |
| **Overall Evaluation Pass Rate** | **100.0%** | **21 / 21 cases passed (100.0%)**. |

---

## 8. Limitations

1. **Static Corpus Scope:** Bounded to `data/policy-manual.md` and `data/Amendment No. 2026-01.md`. Queries on unindexed legal topics are refused based on evidence absence.
2. **Deterministic Conflict Rules:** Conflict detection covers modeled structural contradictions within the policy corpus.
3. **Live Gemini Dependency:** Open-ended generative synthesis requires `GEMINI_API_KEY`; deterministic mock mode is used for offline evaluation.
4. **Non-Legal Advice:** This system is an engineering demonstration and not a substitute for professional legal counsel.

---

## 9. Development Notes

- **Architecture Decisions:** See `DECISIONS.md` for architectural trade-offs, parsing rules, and design rationale.
- **AI Tooling:** See `AI-USAGE.md` for details on AI assistance during development.
- **Policy Integrity:** `data/policy-manual.md` and `data/Amendment No. 2026-01.md` were kept **100% byte-for-byte unchanged** across all phases.
