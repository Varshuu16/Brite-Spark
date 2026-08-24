# BriteSpark Policy Assistant — Problem 1

An evidence-grounded, temporal-aware, conflict-safe policy reasoning engine with deterministic BM25 retrieval, citation verification, and a conversational web interface.

---

## 1. Problem Statement

Administrative policy manuals govern critical public benefit determinations, award calculations, reporting deadlines, and sanction rules. However, deploying Large Language Models directly against raw administrative policies poses critical risks:
- **Hallucinated citations & rules:** Models invent provisions or cite non-existent sections.
- **Temporal confusion:** Policy amendments change deadlines and monetary amounts on specific effective dates; models often fail to apply correct transitional rules (§5.1, §5.2, §5.3).
- **Silent contradiction resolution:** When policies contain conflicting provisions across different chapters, models silently pick one rule without alerting users.
- **Unsafe answers to unindexed topics:** Models attempt to answer out-of-domain questions using general world knowledge rather than refusing based on evidence.

The **BriteSpark Policy Assistant** solves these challenges through a deterministic pipeline combining strict evidence parsing, BM25 retrieval with graph cross-referencing, explicit temporal reasoning, bidirectional citation validation, deterministic conflict detection, and a minimal conversational web interface.

---

## 2. What the System Does

Given any user question (e.g. *"What is the reporting deadline for a change occurring on 15 April 2026?"*), the system:
1. **Parses & Indexes** the official policy manual (`data/policy-manual.md`) and amendments (`data/Amendment No. 2026-01.md`) into 153 discrete, immutable policy clauses.
2. **Extracts Temporal Context** from the query (identifying event dates, determination dates, claim periods, and classification: `PRE_AMENDMENT`, `POST_AMENDMENT`, `SPANNING`, or `UNSPECIFIED`).
3. **Retrieves Policy Evidence** using BM25 ranking augmented by direct clause lookup and cross-reference graph expansion.
4. **Detects Substantive Conflicts** deterministically across retrieved provisions before generation.
5. **Generates Grounded Answers** using Google Gemini with strict prompt constraints, requiring all factual statements to cite retrieved evidence.
6. **Validates Every Citation** against retrieved evidence, rejecting ungrounded or hallucinated citations.
7. **Enforces Refusals** when evidence is insufficient or when questions fall outside the policy scope.
8. **Presents Results** through a conversational web interface, CLI demonstration, and structured JSON API.

---

## 3. Architecture & End-to-End Pipeline

```
                     User Question
                           │
                           ▼
         Policy Parsing / Indexed Policy Corpus
                           │
                           ▼
           BM25 Retrieval + Cross References
                           │
                           ▼
                   Temporal Reasoning
            (Extracts dates & semantic roles)
                           │
                           ▼
                   Conflict Detection
          (Checks substantive contradictions)
                           │
                           ▼
                 Grounded Gemini Answer
            (Strict evidence-only synthesis)
                           │
                           ▼
                  Citation Validation
     (Validates all cited clauses against evidence)
                           │
                           ▼
       Refusal / Completeness / Grounding Checks
                           │
                           ▼
               Flask Conversational UI
```

---

## 4. Pipeline Components

### 4.1 Policy Parsing (`src/parser.py`)
- Deterministic extraction of 148 baseline clauses from `data/policy-manual.md` plus discrete provisions from `data/Amendment No. 2026-01.md` (unified corpus of 153 clauses).
- Preserves titles, numbered sub-clauses `(a)`, `(b)`, lettered paragraphs, and markdown tables (e.g. §6.6.1 and §7.2.1).
- `data/policy-manual.md` remains 100% byte-for-byte immutable.

### 4.2 Deterministic Retrieval (`src/retriever.py`)
- BM25 ranker (`k1=1.5`, `b=0.75`) with token stemming, punctuation stripping, and concept synonym expansion.
- **Direct Citation Boosting:** Queries explicitly mentioning clauses (e.g. *"§4.3.2"*) boost target clauses to rank #1.
- **Cross-Reference Graph Expansion:** Automatically retrieves structurally linked provisions (e.g. calculation formulas §7.1.1 link to allowances §7.2.1; failure to report §10.5.1 links to recipient obligations §4.3.2).

### 4.3 Temporal Engine & Surprise Challenge (`src/temporal.py`)
The Surprise Challenge introduced date-sensitive amendment versioning through **Amendment No. 2026-01** (effective 1 March 2026). The temporal engine resolves date-dependent rules by distinguishing semantic date roles:
- **§5.2 (Change Date Rule):** Reporting deadlines (14 calendar days vs 10 calendar days) are strictly governed by the date the change occurred under §5.2, regardless of when the determination is made.
- **§5.1 (Determination Date Rule):** Earnings disregards ($175 per month vs $120 per month), capital limits, and calculation adjustments apply to determinations made on or after 1 March 2026, even if the underlying claim period is earlier.
- **§5.3 (Spanning Periods):** Claim periods spanning across 1 March 2026 trigger daily rate apportionment under §7.4.3.
- **Unspecified Dates:** When no date is provided, the engine outlines both pre-amendment and post-amendment rules along with their transitional criteria.
- **Temporal Classifications:** Queries are classified into `PRE_AMENDMENT`, `POST_AMENDMENT`, `SPANNING`, or `UNSPECIFIED`. Normal chronological versioning under an amendment is resolved temporally and is **not** treated as an unresolved policy conflict.

### 4.4 Robust Citations & Grounding (`src/answer.py`)
- Regex-based citation extraction capturing canonical clause identifiers (e.g. `§4.3.2`, `Amendment 2026-01 §5.2`, `§10.5.3A`).
- Extracted citations are validated directly against the evidence actually retrieved for that specific question.
- Any unretrieved or hallucinated citations immediately fail validation (`grounded = False`, `unsupported_citations = [...]`).
- Filter rejects numeric false positives (e.g. plain numbers, dates, currency values like `$175`, and day counts like `10 calendar days` are not treated as citations).

### 4.5 Refusal & Safety Contract (`src/answer.py`)
- Fast refusal path before invoking Gemini when retrieved evidence is absent, insufficient, or completely out-of-domain (e.g. *"What is the capital of France?"*).
- Refusal contract guarantees:
  - `grounded = False`
  - `insufficient_evidence = True`
  - `citations = []`

### 4.6 Substantive Conflict Detection (`src/conflict.py`)
- Detects genuine substantive contradictions across retrieved evidence where provisions describe conflicting rules that cannot be resolved merely by date (e.g. the 10/14-day recipient reporting rule under §4.3.2 vs the 30-day overpayment notification reference under §9.1.4).
- Distinguishes chronological amendment updates (which are resolved by the temporal engine) from true substantive contradictions.
- The system does not silently pick a single rule; it explicitly highlights the contradiction, cites both conflicting clauses, and presents the context clearly.

---

## 5. Web Interface (Part 8)

The web UI provides a clean, minimal conversational interface built with Flask:
- **Clean Initial Home State:** A centered layout with a white background, "BriteSpark Policy Assistant" heading, simple subtitle, rounded search input, and quick-fill example question pills. No decorative AI gradients, glowing borders, or clutter.
- **Conversational Transition:** Upon asking the first question, the interface transitions into a conversation-style stream. Previous questions and answers remain visible during the active session.
- **Answer Presentation:** Each assistant response cleanly presents:
  - Status tag (`[ Grounded ]` or `[ Refusal / Insufficient Evidence ]`)
  - Grounded answer text
  - Status metadata row (`Grounding Status`, `Evidence Sufficiency`, `Temporal Status`, `Conflict Status`)
  - Structured **Sources & Validated Citations** cards detailing clause IDs, source files, section titles, and verbatim quotes.
- **Refusal & Conflict Handling:** Refusal states display a clear notice indicating insufficient evidence. Detected conflicts display a warning banner citing the conflicting provisions without assuming precedence.
- **Interactive Multi-Turn Queries:** A bottom query input allows follow-up questions to be asked in the same conversation.
- **Reset & Session Management:** A `+ New Question` button resets the session back to the clean initial home state.

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Installation
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
*Note: If `GEMINI_API_KEY` is not configured, the web UI, evaluation suite, and unit tests run in deterministic offline mode.*

---

## 7. Running the Application

### 7.1 Start the Web UI
```bash
python app.py
```
Open your browser at **http://127.0.0.1:5000**

### 7.2 Run CLI Demonstration
```bash
python src/answer.py
```

### 7.3 Run Full Regression Test Suite (109 Tests)
```bash
python -m unittest discover tests -v
```
- **109 total tests**
- **109 passed** (0 failures, 0 errors)
- **2 skipped** live network socket tests when `GEMINI_API_KEY` is unavailable

### 7.4 Run Deterministic Multi-Dimensional Evaluation (21 Cases)
```bash
python tests/run_evaluation.py
```

---

## 8. Evaluation Metrics

The deterministic evaluation framework (`tests/evaluation_dataset.py`, `tests/run_evaluation.py`) measures 6 quality dimensions across 21 test scenarios:

| Metric Dimension | Score | Description |
| :--- | :--- | :--- |
| **Evidence Retrieval Accuracy** | **100.0%** | Expected governing clauses retrieved in top-$k$ results. |
| **Temporal Classification Accuracy** | **100.0%** | Exact categorization into `PRE_AMENDMENT`, `POST_AMENDMENT`, `SPANNING`, or `UNSPECIFIED`. |
| **Citation Validity Rate** | **95.2%** | Validated citations strictly match retrieved evidence; unretrieved citations are flagged. |
| **Citation Completeness Rate** | **95.2%** | Answers with substantive assertions require citations; un-cited assertions fail. |
| **Refusal Safety Accuracy** | **100.0%** | Immediate structured refusal for out-of-domain and zero-evidence queries. |
| **Conflict Detection Accuracy** | **100.0%** | Substantive contradictions are flagged while temporal versioning is correctly resolved. |
| **Overall Evaluation Pass Rate** | **100.0%** | 21 / 21 evaluation scenarios passing. |

---

## 9. System Limitations

1. **Static Corpus Boundary:** The engine is strictly bounded to `data/policy-manual.md` and `data/Amendment No. 2026-01.md`. Questions requiring external statutes or unindexed topics are refused based on evidence absence.
2. **Deterministic Conflict Patterns:** Conflict detection is deterministic and covers modeled structural contradictions within the policy corpus.
3. **Live Gemini Dependency:** Live open-ended natural language generation requires `GEMINI_API_KEY`; deterministic mock mode is used for offline evaluation.
4. **Non-Legal Advice:** This system is an engineering demonstration for grounded policy question-answering and is not a substitute for formal legal review.

---

## 10. AI Disclosure & Engineering Notes

- AI tools were used during development for learning, brainstorming, implementation assistance, debugging, test development, and documentation. The final implementation was reviewed and tested as part of the project.
- For details, see `AI-USAGE.md`.
- `data/policy-manual.md` was preserved **100% byte-for-byte unchanged** throughout all phases of development.
