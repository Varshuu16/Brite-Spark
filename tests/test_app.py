"""
Unit tests for the BriteSpark Policy Assistant Flask Web Application (Part 8).
"""

import unittest
from unittest.mock import patch
import uuid
import app as app_module
from app import app


class TestWebApplication(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        app.config["TESTING"] = True

    def test_session_cookie_is_non_permanent(self):
        """Verify Flask session is configured as a non-permanent browser session."""
        self.assertFalse(app.config.get("SESSION_PERMANENT", True))
        self.assertTrue(app.config.get("SESSION_COOKIE_HTTPONLY", False))

    def test_get_index_returns_200_and_brand_title(self):
        """Verify GET / returns HTTP 200 and renders the clean initial hero without error notices."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("BriteSpark Policy Assistant", content)
        self.assertIn("Ask anything about the policy manual", content)
        self.assertIn("Ask Policy", content)
        self.assertNotIn("Please enter a policy question to search.", content)

    def test_post_empty_question_handled_safely(self):
        """Verify submitting an empty question returns a clear notice without crashing."""
        response = self.client.post("/", data={"question": ""})
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Please enter a policy question", content)

    def test_post_grounded_question_renders_answer_and_citations(self):
        """Verify asking a valid policy question renders the grounded answer and sources."""
        response = self.client.post(
            "/",
            data={"question": "What is the earnings disregard?"},
        )
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Answer", content)
        self.assertIn("Grounding Status", content)
        self.assertIn("Sources &amp; Validated Citations", content)
        self.assertIn("6.4.1", content)

    def test_post_out_of_domain_question_renders_refusal_state(self):
        """Verify asking an out-of-domain question displays the insufficient evidence banner."""
        response = self.client.post(
            "/",
            data={"question": "What is the capital of France?"},
        )
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Insufficient Policy Evidence", content)
        self.assertIn("The available policy evidence does not contain sufficient facts", content)

    def test_post_conflict_question_renders_conflict_banner(self):
        """Verify questions with conflicting evidence display the conflict banner."""
        response = self.client.post(
            "/",
            data={"question": "What is the deadline for reporting a change of circumstances?"},
        )
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Potential Policy Conflict Detected", content)
        self.assertIn("CONF-REPORTING-01", content)

    def test_conversation_history_retention_in_active_session(self):
        """Verify multi-turn questioning preserves previous question and answer during active session."""
        # First turn
        self.client.post("/", data={"question": "What is the earnings disregard?"})
        # Second turn
        response = self.client.post("/", data={"question": "A change occurred on 15 April 2026. What is the reporting deadline?"})
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        # Both turns should appear in the active session
        self.assertIn("What is the earnings disregard?", content)
        self.assertIn("A change occurred on 15 April 2026. What is the reporting deadline?", content)
        self.assertIn("14 calendar days", content)

    def test_clear_conversation_resets_to_home_state(self):
        """Verify GET /clear resets conversation history back to the clean initial home state."""
        self.client.post("/", data={"question": "What is the earnings disregard?"})
        response = self.client.get("/clear")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Ask anything about the policy manual", content)
        self.assertNotIn("class=\"user-bubble\"", content)

    def test_server_restart_generation_id_clears_old_session(self):
        """Verify that when server restarts (new SERVER_GENERATION_ID), old session history is invalidated."""
        # Ask a question during generation 1
        self.client.post("/", data={"question": "What is the earnings disregard?"})

        # Simulate server restart by generating a new SERVER_GENERATION_ID
        original_gen = app_module.SERVER_GENERATION_ID
        try:
            app_module.SERVER_GENERATION_ID = str(uuid.uuid4())

            # Next request with the existing client/cookie
            response = self.client.get("/")
            self.assertEqual(response.status_code, 200)
            content = response.data.decode("utf-8")

            # Must render the clean initial home state without previous user bubble
            self.assertIn("Ask anything about the policy manual", content)
            self.assertNotIn("class=\"user-bubble\"", content)
        finally:
            app_module.SERVER_GENERATION_ID = original_gen

    def test_new_browser_session_starts_empty(self):
        """Verify a fresh client without cookies starts with an empty initial state."""
        new_client = app.test_client()
        response = new_client.get("/")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Ask anything about the policy manual", content)

    def test_api_ask_returns_structured_json(self):
        """Verify POST /api/ask returns structured AnswerResult dictionary."""
        response = self.client.post(
            "/api/ask",
            json={"question": "A change occurred on 15 April 2026. What is the reporting deadline?"},
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("question", json_data)
        self.assertIn("result", json_data)
        res = json_data["result"]
        self.assertTrue(res["grounded"])
        self.assertEqual(res["temporal_status"], "POST_AMENDMENT")
        self.assertIn("4.3.2", res["citations"])

    def test_api_ask_empty_question_returns_400(self):
        """Verify POST /api/ask with empty question returns HTTP 400 Bad Request."""
        response = self.client.post("/api/ask", json={"question": "  "})
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("error", json_data)

    def test_exception_handling_does_not_expose_stack_trace(self):
        """Verify unexpected backend errors are caught and shown gracefully without stack traces."""
        with patch("app.generate_answer", side_effect=RuntimeError("Database socket error")):
            response = self.client.post("/", data={"question": "What is the policy?"})
            self.assertEqual(response.status_code, 200)
            content = response.data.decode("utf-8")
            self.assertIn("An error occurred while processing your request", content)
            self.assertNotIn("Traceback (most recent call last)", content)


if __name__ == "__main__":
    unittest.main()
