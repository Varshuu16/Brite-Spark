# BriteSpark Policy Assistant — Problem 1

An evidence-grounded, temporal-aware, conflict-safe policy reasoning engine with deterministic BM25 retrieval, citation verification, and a responsive web interface.

---

## 1. Problem Statement

Administrative policy manuals govern critical public benefit determinations, award calculations, reporting deadlines, and sanction rules. However, deploying Large Language Models directly against raw administrative policies poses critical risks:
- **Hallucinated citations & rules:** Models invent provisions or cite non-existent sections.
- **Temporal confusion:** Policy amendments change deadlines and monetary amounts on specific effective dates; models often fail to apply correct transitional rules (§5.1, §5.2, §5.3).
- **Silent contradiction resolution:** When policies contain conflicting provisions across different chapters, models silently pick one rule without alerting users.
- **Unsafe answers to unindexed topics:** Models attempt to answer out-of-domain questions using general world knowledge rather than refusing based on evidence.

The **BriteSpark Policy Assistant** solves these challenges through a deterministic pipeline combining strict evidence parsing, BM25 retrieval with graph cross-referencing, explicit temporal reasoning, bidirectional citation validation, deterministic conflict detection, and a clean web interface.

---

## 2. What the System Does

Given any user question (e.g. *"What is the reporting deadline for a change occurring on 15 April 2026?"*), the system:
1. **Parses & Indexes** the official policy manual (`data/policy-manual.md`) and amendments (`data/Amendment No. 2026-01.md`) into 153 discrete, immutable policy clauses.
2. **Extracts Temporal Context** from the query (determining event dates, determination dates, claim periods, and classification: `PRE_AMENDMENT`, `POST_AMENDMENT`, `SPANNING`, or `UNSPECIFIED`).
3. **Retrieves Policy Evidence** using BM25 ranking augmented by direct clause lookup and cross-reference expansion.
4. **Detects Substantive Conflicts** deterministically across retrieved provisions before generation.
5. **Generates Grounded Answers** using Google Gemini with strict prompt constraints, requiring all factual statements to cite retrieved evidence.
6. **Validates Every Citation** against retrieved evidence, rejecting ungrounded or hallucinated citations.
7. **Enforces Refusals** when evidence is insufficient or when questions fall outside the policy scope.
8. **Presents Results** through a web interface, CLI demonstration, and JSON API.

---

## 3. Architecture

```
                      User Question
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      Temporal Engine            BM25 Policy Retriever
   (Extracts dates & rules)   (Scores clauses + cross-refs)
               │                         │
               └────────────┬────────────┘
                            ▼
                Conflict Detection Engine
              (Checks substantive contradictions)
                            │
                            ▼
              Grounded Gemini Generation
             (Strict evidence-only synthesis)
                            │
                            ▼
               Citation Verification Engine
          (Validates all cited clauses against evidence)
                            │
                            ▼
               Structured AnswerResult
    (Answer • Status • Validated Sources • Conflicts)
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

### 4.3 Temporal Engine & Amendment No. 2026-01 (`src/temporal.py`)
Amendment No. 2026-01 took effect on **1 March 2026**. The engine implements strict transitional logic:
- **§5.1 (Determination Date Rule):** Earnings disregard ($175 vs $120), capital limits, and sanctions apply to determinations made on/after 1 March 2026.
- **§5.2 (Change Date Rule):** Reporting deadlines (14 days vs 10 days) are controlled strictly by the date the change occurred, regardless of determination date.
- **§5.3 (Spanning Periods):** Claim periods spanning across 1 March 2026 trigger daily apportionment under §7.4.3.
- **Unspecified Dates:** Fully explains both pre-amendment and post-amendment rules with transitional criteria.

### 4.4 Robust Citations & Grounding (`src/answer.py`)
- Regex-based citation extraction supporting canonical clause identifiers (`§4.3.2`, `Amendment 2026-01 §5.2`).
- Filter rejects numeric false positives (e.g. `$175`, `10 calendar days`, `2026`).
- Unretrieved or hallucinated citations immediately fail grounding (`grounded = False`, `unsupported_citations = [...]`).

### 4.5 Refusal & Safety Contract (`src/answer.py`)
- Zero-evidence fast path before LLM invocation for out-of-domain questions (e.g. *"What is the capital of France?"*).
- Refusal contract guarantees: `grounded = False`, `insufficient_evidence = True`, `citations = []`.

### 4.6 Substantive Conflict Detection (`src/conflict.py`)
- Detects contradictions across retrieved evidence (e.g. general 10/14-day reporting timeframe §4.3.2 vs 30-day overpayment notification §9.1.4).
- Distinguishes chronological amendment evolution from genuine substantive conflicts.

---

## 5. Web Interface (Part 8)

The Flask web UI provides a clean, responsive interface:
- **Header & Badges:** Grounded • Evidence-Based • Temporal-Aware • Conflict-Safe.
- **Question Box:** Large textarea with quick-fill example chips.
- **Status Dashboard:** Grounding Status, Evidence Sufficiency, Temporal Classification, and Conflict Status.
- **Answer Display:** Formatted grounded response.
- **Validated Sources Card:** List of cited clauses with source document, clause titles, and verbatim text excerpts.
- **Refusal & Conflict Banners:** Clear visual alerts for insufficient evidence or detected conflicts.

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

### 7.3 Run Full Regression Test Suite (107 Tests)
```bash
python -m unittest discover tests -v
```

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

1. **Static Corpus Boundary:** Only indexes provisions contained in `data/policy-manual.md` and `data/Amendment No. 2026-01.md`. Questions requiring external legal statutes will be refused as insufficient evidence.
2. **Lexical BM25 Search:** Rare, highly abstract conceptual synonyms not covered in synonym maps may receive lower BM25 ranks unless cross-referenced.
3. **Offline Mock Scope:** Offline mock generator covers the primary benchmark and evaluation queries; live generative questioning requires `GEMINI_API_KEY`.

---

## 10. AI Disclosure & Engineering Notes

- Architecture, parsing strategies, temporal resolution rules, citation validation algorithms, conflict detection matrices, and testing frameworks were designed and implemented specifically for the BriteSpark Problem 1 Challenge.
- All evaluation metrics are deterministic and reproducible offline without live API dependencies.
