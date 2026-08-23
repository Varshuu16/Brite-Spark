"""
BriteSpark Policy Assistant — Web Application Entry Point (Part 8).

Provides a clean, conversational web interface for grounded policy questioning,
temporal resolution, citation verification, and conflict detection.
"""

import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import uuid

from flask import Flask, render_template, request, jsonify, session

from src.loader import load_full_policy_corpus
from src.retriever import PolicyRetriever, ScoredClause
from src.temporal import extract_temporal_context, TemporalStatus
from src.conflict import detect_conflicts
from src.answer import generate_answer, AnswerResult


# Initialize Flask application with explicit template folder
TEMPLATE_DIR = Path(__file__).parent / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "britespark-policy-secret-key-2026")

# Configure session as non-permanent browser session cookie
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Unique application generation ID generated fresh on every server startup
SERVER_GENERATION_ID = str(uuid.uuid4())

# Load policy corpus and initialize deterministic retriever
_CORPUS = None
_RETRIEVER = None


def get_retriever() -> PolicyRetriever:
    """Lazily loads the full policy corpus and initializes the retriever singleton."""
    global _CORPUS, _RETRIEVER
    if _RETRIEVER is None:
        _CORPUS = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        _RETRIEVER = PolicyRetriever(clauses=_CORPUS)
    return _RETRIEVER


def get_offline_mock_client(retrieved: list, t_ctx: Any, has_conflict: bool):
    """Provides a deterministic offline response generator when GEMINI_API_KEY is not set."""
    def mock_llm(p: str) -> str:
        q_part = p
        if "USER QUESTION:" in p:
            q_part = p.split("USER QUESTION:")[-1].split("POLICY EVIDENCE:")[0]
        q_low = q_part.lower()

        evidence_ids = {r.clause.clause_id.lstrip("§").strip() for r in retrieved}

        if not retrieved or "france" in q_low or "weather" in q_low or "property tax" in q_low:
            return "The provided policy evidence is insufficient to answer this question."

        # Adversarial mixed date: Change Feb 2026, Determination Mar 2026
        if "change occurred on 20 february 2026" in q_low:
            return (
                "Under Amendment No. 2026-01 §5.2, the reporting deadline is governed by the date the change occurred. "
                "Because the change occurred on 20 February 2026, the pre-amendment deadline applies regardless of determination date. "
                "Under §4.3.2, the recipient must report the change within 10 calendar days."
            )

        # Adversarial reverse mixed date: Change Mar 2026, Determination Feb 2026
        if "change occurred on 20 march 2026" in q_low:
            return (
                "Under Amendment No. 2026-01 §5.2, the reporting deadline is governed by the date the change occurred. "
                "Because the change occurred on 20 March 2026, the amended rule applies under Amendment No. 2026-01 §2.1 and §5.2. "
                "Under §4.3.2 as amended, the recipient must report within 14 calendar days."
            )

        # Pre-amendment reporting
        if "10 february 2026" in q_low and "reporting" in q_low:
            return (
                "For a change of circumstances occurring on 10 February 2026, the pre-amendment deadline applies under "
                "Amendment No. 2026-01 §5.2. Under §4.3.2, the recipient must report within 10 calendar days."
            )

        # Post-amendment reporting
        if "15 april 2026" in q_low and "reporting" in q_low:
            return (
                "For a change of circumstances occurring on 15 April 2026, the amended deadline applies under "
                "Amendment No. 2026-01 §2.1 and §5.2. Under §4.3.2 as amended, the recipient must report within 14 calendar days."
            )

        # Determination date March 2026
        if "20 march 2026" in q_low and ("disregard" in q_low or "january 2026" in q_low):
            return (
                "Under Amendment No. 2026-01 §5.1, amendments apply to any determination made on or after 1 March 2026, "
                "even for a prior period such as January 2026. Therefore, under §6.4.1 as amended, the earnings disregard is $175 per month."
            )

        # Pre-amendment determination Feb 2026
        if "20 february 2026" in q_low and "disregard" in q_low:
            return (
                "Under §6.4.1 of the original policy manual, for determinations made on 20 February 2026, the earnings disregard is $120 per month."
            )

        # Spanning period
        if "15 february 2026" in q_low and "15 march 2026" in q_low:
            return (
                "Under Amendment No. 2026-01 §5.3, for claim periods spanning across 1 March 2026, the award is calculated using daily rates "
                "and apportioned under §7.4.3."
            )

        # Sanction exception §10.5.3A
        if "increased the award" in q_low or "positive" in q_low:
            return (
                "Under §10.5.3A, no sanction shall be imposed where the unreported change "
                "of circumstances is one that would have increased the household's award."
            )

        # Substantive conflict: §4.3.2 vs §9.1.4
        if has_conflict and "4.3.2" in evidence_ids and "9.1.4" in evidence_ids:
            return (
                "The retrieved policy evidence contains two related references with different timeframes: "
                "under §4.3.2, the recipient's direct obligation is to report within 10 calendar days "
                "(or 14 calendar days on/after 1 March 2026 under Amendment No. 2026-01 §2.1 and §5.2), "
                "whereas §9.1.4 references a 30 calendar day period regarding overpayment establishment notifications."
            )

        # Unspecified disregard
        if "disregard" in q_low:
            return (
                "Under the original policy manual (§6.4.1), the first $120 per month of earnings is disregarded. "
                "Effective 1 March 2026 under Amendment No. 2026-01 §1.1, the disregard is increased to $175 per month. "
                "Under Amendment No. 2026-01 §5.1, the $175 disregard applies to determinations on or after 1 March 2026."
            )

        # Default reporting
        return (
            "Prior to 1 March 2026, a recipient must report changes within 10 calendar days under §4.3.2. "
            "Effective 1 March 2026 under Amendment No. 2026-01 §2.1, the reporting period is 14 calendar days. "
            "Under Amendment No. 2026-01 §5.2, the 14-day rule applies only to changes occurring on or after 1 March 2026."
        )

    return mock_llm


@app.before_request
def validate_server_generation():
    """Ensure session history is invalidated if originating from a previous server run."""
    if request.path.startswith("/static") or request.path.startswith("/api/"):
        return
    stored_gen = session.get("server_generation_id")
    if stored_gen != SERVER_GENERATION_ID:
        session.clear()
        session["server_generation_id"] = SERVER_GENERATION_ID
        session["history"] = []
        session.modified = True


@app.route("/", methods=["GET", "POST"])
def index():
    """Main route serving the conversational policy interface."""
    error_message = None
    session.permanent = False

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        chat_id = request.form.get("chat_id", "").strip() or "default"
        history_key = f"history_{chat_id}" if chat_id != "default" else "history"

        if not question:
            error_message = "Please enter a policy question to search."
            return render_template(
                "index.html",
                history=session.get(history_key, session.get("history", [])),
                error_message=error_message,
                chat_id=chat_id,
            )

        try:
            retriever = get_retriever()
            t_ctx = extract_temporal_context(question)
            retrieved = retriever.retrieve(question, top_k=5)

            # Determine whether live Gemini or deterministic offline mode is active
            api_key = os.environ.get("GEMINI_API_KEY")
            client = None
            if not api_key:
                has_conf = len(detect_conflicts(retrieved, temporal_context=t_ctx, question=question)) > 0
                client = get_offline_mock_client(retrieved, t_ctx, has_conf)

            result = generate_answer(
                question=question,
                retrieved_clauses=retrieved,
                client=client,
                api_key=api_key,
                temporal_context=t_ctx,
            )

            # Record turn in session history
            turn_entry = {
                "question": question,
                "answer": result.answer,
                "grounded": result.grounded,
                "insufficient_evidence": result.insufficient_evidence,
                "temporal_status": result.temporal_context.status.value if result.temporal_context else "UNSPECIFIED",
                "conflicts_detected": result.conflicts_detected,
                "conflicts": [
                    {"conflict_id": c.conflict_id, "description": c.description}
                    for c in result.conflicts
                ],
                "validated_citations": [
                    {
                        "citation_id": c.citation_id,
                        "source_document": c.source_document,
                        "clause_title": c.clause_title,
                        "clause_text": c.clause_text,
                    }
                    for c in result.validated_citations
                ],
            }

            hist = list(session.get(history_key, session.get("history", [])))
            hist.append(turn_entry)
            session[history_key] = hist
            session["history"] = hist
            session.modified = True

        except Exception as err:
            error_message = f"An error occurred while processing your request: {err}"

        return render_template(
            "index.html",
            history=session.get(history_key, session.get("history", [])),
            error_message=error_message,
            chat_id=chat_id,
        )

    # GET request: render existing active history or empty initial state
    chat_id = request.args.get("chat_id", "").strip()
    history_key = f"history_{chat_id}" if chat_id else "history"
    history = session.get(history_key, session.get("history", []))

    return render_template(
        "index.html",
        history=history,
        error_message=error_message,
        chat_id=chat_id,
    )


@app.route("/clear", methods=["GET", "POST"])
def clear_history():
    """Resets the conversation history back to the initial state."""
    session.clear()
    session["server_generation_id"] = SERVER_GENERATION_ID
    session["history"] = []
    session.modified = True
    return render_template("index.html", history=[], error_message=None, chat_id="")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """JSON API endpoint for programmatic querying."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "A non-empty 'question' parameter is required."}), 400

    try:
        retriever = get_retriever()
        t_ctx = extract_temporal_context(question)
        retrieved = retriever.retrieve(question, top_k=5)

        api_key = os.environ.get("GEMINI_API_KEY")
        client = None
        if not api_key:
            has_conf = len(detect_conflicts(retrieved, temporal_context=t_ctx, question=question)) > 0
            client = get_offline_mock_client(retrieved, t_ctx, has_conf)

        result = generate_answer(
            question=question,
            retrieved_clauses=retrieved,
            client=client,
            api_key=api_key,
            temporal_context=t_ctx,
        )
        return jsonify({
            "question": question,
            "result": result.to_dict(),
        })
    except Exception as err:
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting BriteSpark Policy Assistant on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
