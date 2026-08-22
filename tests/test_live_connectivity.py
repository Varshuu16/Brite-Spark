"""
Live Gemini API connectivity and diagnostic test.
This test checks model availability and socket latency when GEMINI_API_KEY is configured.
If GEMINI_API_KEY is not set, this test skips cleanly without failing the suite.
"""

import os
import time
import unittest

try:
    from google import genai
    from google.genai import types
    HAVE_GEMINI_SDK = True
except ImportError:
    HAVE_GEMINI_SDK = False

from src.answer import DEFAULT_GEMINI_MODEL, DEFAULT_TIMEOUT_SECONDS


class TestLiveGeminiConnectivity(unittest.TestCase):
    """Diagnostic suite for checking live Gemini endpoint and model reachability."""

    def setUp(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY environment variable is not configured; skipping live network test.")
        if not HAVE_GEMINI_SDK:
            self.skipTest("google-genai SDK is not installed; skipping live network test.")

    def test_minimal_ping(self):
        """Send a minimal 'Reply with OK' prompt to test socket connectivity and response time."""
        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=int(DEFAULT_TIMEOUT_SECONDS * 1000)),
        )

        model_to_test = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        start_time = time.time()

        try:
            response = client.models.generate_content(
                model=model_to_test,
                contents="Reply with OK",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    http_options=types.HttpOptions(timeout=int(DEFAULT_TIMEOUT_SECONDS * 1000)),
                ),
            )
            elapsed = time.time() - start_time
            self.assertTrue(response.text)
            print(f"\n[Live API Diagnostic] Minimal ping with '{model_to_test}' succeeded in {elapsed:.2f}s: {response.text.strip()[:60]}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[Live API Diagnostic] Minimal ping with '{model_to_test}' failed after {elapsed:.2f}s: {type(e).__name__}: {e}")
            raise

    def test_full_grounded_answer_live(self):
        """Test full answer generation pipeline end-to-end with live Gemini API."""
        from src.loader import load_policy
        from src.retriever import PolicyRetriever
        from src.answer import generate_answer

        clauses = load_policy("data/policy-manual.md")
        retriever = PolicyRetriever(clauses=clauses)

        question = "What is the deadline for reporting a change of circumstances?"
        retrieved = retriever.retrieve(question, top_k=5)

        start_time = time.time()
        result = generate_answer(question, retrieved, api_key=self.api_key)
        elapsed = time.time() - start_time

        print(f"\n[Live Pipeline Test] Completed in {elapsed:.2f}s:")
        print(f"Answer: {result.answer[:120]}...")
        print(f"Citations: {result.citations}")
        print(f"Grounded: {result.grounded}")

        self.assertTrue(result.grounded)
        self.assertFalse(result.insufficient_evidence)
        self.assertIn("4.3.2", result.citations)


if __name__ == "__main__":
    unittest.main()
