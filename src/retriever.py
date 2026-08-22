"""
Deterministic Lexical Policy Retriever using BM25 with multi-field weighting,
Porter stemming, exact phrase boosting, and cross-reference expansion.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from .models import PolicyClause
    from .loader import load_policy
except ImportError:
    from models import PolicyClause
    from loader import load_policy


# Standard English stopwords to filter out non-informative noise words
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class PorterStemmer:
    """
    Standard deterministic implementation of the Martin Porter stemming algorithm.
    """

    def is_consonant(self, word: str, i: int) -> bool:
        letter = word[i]
        if letter in "aeiou":
            return False
        if letter == "y":
            if i == 0:
                return True
            return not self.is_consonant(word, i - 1)
        return True

    def get_measure(self, word: str) -> int:
        form = []
        for i in range(len(word)):
            form.append("c" if self.is_consonant(word, i) else "v")
        form_str = "".join(form)
        return len(re.findall(r"vc", form_str))

    def contains_vowel(self, word: str) -> bool:
        for i in range(len(word)):
            if not self.is_consonant(word, i):
                return True
        return False

    def ends_with_double_consonant(self, word: str) -> bool:
        if len(word) < 2:
            return False
        return word[-1] == word[-2] and self.is_consonant(word, len(word) - 1)

    def cvc(self, word: str) -> bool:
        if len(word) < 3:
            return False
        return (
            self.is_consonant(word, len(word) - 3)
            and not self.is_consonant(word, len(word) - 2)
            and self.is_consonant(word, len(word) - 1)
            and word[-1] not in "wxy"
        )

    def stem(self, word: str) -> str:
        w = word.lower()
        if len(w) <= 2:
            return w

        # Common policy root normalization aliases
        if w in {"apply", "applies", "applicant", "applicants", "application", "applications", "applicable"}:
            return "applic"
        if w in {"eligible", "eligibility", "ineligible", "ineligibility"}:
            return "eligib"
        if w in {"reside", "residence", "resident", "residential", "residing"}:
            return "resid"

        # Step 1a
        if w.endswith("sses"):
            w = w[:-2]
        elif w.endswith("ies"):
            w = w[:-2]
        elif w.endswith("ss"):
            pass
        elif w.endswith("s"):
            w = w[:-1]

        # Step 1b
        flag = False
        if w.endswith("eed"):
            stem = w[:-3]
            if self.get_measure(stem) > 0:
                w = stem + "ee"
        elif w.endswith("ed") and self.contains_vowel(w[:-2]):
            w = w[:-2]
            flag = True
        elif w.endswith("ing") and self.contains_vowel(w[:-3]):
            w = w[:-3]
            flag = True

        if flag:
            if w.endswith("at") or w.endswith("bl") or w.endswith("iz"):
                w = w + "e"
            elif self.ends_with_double_consonant(w) and not (
                w.endswith("l") or w.endswith("s") or w.endswith("z")
            ):
                w = w[:-1]
            elif self.get_measure(w) == 1 and self.cvc(w):
                w = w + "e"

        # Step 1c
        if w.endswith("y") and self.contains_vowel(w[:-1]):
            w = w[:-1] + "i"

        # Step 2
        step2_pairs = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
            ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
            ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
            ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
        ]
        for suffix, repl in step2_pairs:
            if w.endswith(suffix):
                stem = w[:-len(suffix)]
                if self.get_measure(stem) > 0:
                    w = stem + repl
                break

        # Step 3
        step3_pairs = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
            ("ical", "ic"), ("ful", ""), ("ness", ""),
        ]
        for suffix, repl in step3_pairs:
            if w.endswith(suffix):
                stem = w[:-len(suffix)]
                if self.get_measure(stem) > 0:
                    w = stem + repl
                break

        # Step 4
        step4_suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
            "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ]
        for suffix in step4_suffixes:
            if w.endswith(suffix):
                stem = w[:-len(suffix)]
                if self.get_measure(stem) > 1:
                    w = stem
                break
        if w.endswith("ion"):
            stem = w[:-3]
            if self.get_measure(stem) > 1 and (stem.endswith("s") or stem.endswith("t")):
                w = stem

        # Step 5a
        if w.endswith("e"):
            stem = w[:-1]
            m = self.get_measure(stem)
            if m > 1 or (m == 1 and not self.cvc(stem)):
                w = stem

        # Step 5b
        if self.get_measure(w) > 1 and self.ends_with_double_consonant(w) and w.endswith("l"):
            w = w[:-1]

        return w


STEMMER = PorterStemmer()


def tokenize(text: str, stem: bool = True, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenizes, normalizes, and stems raw text into a list of searchable terms.
    """
    if not text:
        return []
    # Tokenize words, numbers, and section markers
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_\.\$]+\b", text.lower())
    tokens = []
    for token in raw_tokens:
        clean = token.strip("._§")
        if not clean:
            continue
        if remove_stopwords and clean in STOP_WORDS:
            continue
        if stem:
            clean = STEMMER.stem(clean)
        tokens.append(clean)
    return tokens


@dataclass
class ScoredClause:
    """
    A policy clause paired with its computed relevance score.
    """
    clause: PolicyClause
    score: float

    @property
    def clause_id(self) -> str:
        return self.clause.clause_id

    @property
    def citation(self) -> str:
        return self.clause.citation

    @property
    def clause_text(self) -> str:
        return self.clause.clause_text

    @property
    def clause_title(self) -> Optional[str]:
        return self.clause.clause_title

    @property
    def parent_section(self) -> Optional[str]:
        return self.clause.parent_section

    @property
    def parent_part(self) -> Optional[str]:
        return self.clause.parent_part

    def to_dict(self) -> Dict[str, Any]:
        data = self.clause.to_dict()
        data["score"] = round(self.score, 4)
        return data

    def __str__(self) -> str:
        title_info = f" ({self.clause_title})" if self.clause_title else ""
        return f"[{self.citation}{title_info} | Score: {self.score:.4f}] {self.clause_text[:100]}..."


class PolicyRetriever:
    """
    Deterministic Lexical BM25 retriever with multi-field weighting and exact phrase scoring.
    """

    def __init__(
        self,
        clauses: Optional[List[PolicyClause]] = None,
        policy_path: Union[str, Path] = "data/policy-manual.md",
        k1: float = 1.2,
        b: float = 0.75,
        min_score: float = 0.5,
    ):
        """
        Initializes the retriever and indexes the supplied clauses.
        
        Args:
            clauses: Optional pre-loaded list of PolicyClause objects.
            policy_path: Path to load policy clauses from if clauses not provided.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            min_score: Default minimum relevance threshold for returning results.
        """
        if clauses is not None:
            self.clauses = clauses
        else:
            self.clauses = load_policy(policy_path)

        self.k1 = k1
        self.b = b
        self.min_score = min_score

        # Field weighting configuration
        self.weights = {
            "title": 4.0,
            "section": 3.0,
            "part": 1.0,
            "text": 1.5,
        }

        self._build_index()

    def _build_index(self):
        """Builds inverted index, document token statistics, and IDF tables."""
        self.doc_count = len(self.clauses)
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[float] = []
        self.df: Dict[str, int] = {}
        self.clause_index: Dict[str, int] = {}

        # Raw texts for exact phrase matching
        self.raw_texts: List[str] = []

        total_length = 0.0

        for idx, clause in enumerate(self.clauses):
            self.clause_index[clause.clause_id] = idx

            # Tokenize individual fields
            title_tokens = tokenize(clause.clause_title or "", stem=True, remove_stopwords=True)
            section_tokens = tokenize(clause.parent_section or "", stem=True, remove_stopwords=True)
            part_tokens = tokenize(clause.parent_part or "", stem=True, remove_stopwords=True)
            text_tokens = tokenize(clause.clause_text, stem=True, remove_stopwords=True)

            # Combined weighted token sequence for term frequencies
            combined_tokens = []
            combined_tokens.extend(title_tokens * int(self.weights["title"]))
            combined_tokens.extend(section_tokens * int(self.weights["section"]))
            combined_tokens.extend(part_tokens * int(self.weights["part"]))
            combined_tokens.extend(text_tokens * int(self.weights["text"]))

            doc_len = len(combined_tokens)
            self.doc_tokens.append(combined_tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            # Track unique terms for Document Frequency (DF)
            unique_terms = set(combined_tokens)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1

            # Store full lowercased text for exact substring / phrase matching
            full_text = f"{clause.parent_part or ''} {clause.parent_section or ''} {clause.clause_title or ''} {clause.clause_text}".lower()
            self.raw_texts.append(full_text)

        self.avg_doc_len = total_length / self.doc_count if self.doc_count > 0 else 1.0

        # Calculate BM25 IDF for all indexed terms
        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            val = (self.doc_count - freq + 0.5) / (freq + 0.5)
            self.idf[term] = math.log(1.0 + max(val, 0.01)) + 1.0

    def _compute_exact_phrase_bonus(self, query_raw: str, doc_idx: int) -> float:
        """
        Computes an additional bonus score if consecutive query words appear as an exact phrase.
        """
        raw_doc = self.raw_texts[doc_idx]
        bonus = 0.0

        words = re.findall(r"\b[a-zA-Z0-9_\$]+\b", query_raw.lower())
        non_stop_words = [w for w in words if w not in STOP_WORDS]

        # Bigram exact matches
        if len(non_stop_words) >= 2:
            for i in range(len(non_stop_words) - 1):
                bigram = f"{non_stop_words[i]} {non_stop_words[i+1]}"
                if bigram in raw_doc:
                    bonus += 3.0

        # Trigram exact matches
        if len(non_stop_words) >= 3:
            for i in range(len(non_stop_words) - 2):
                trigram = f"{non_stop_words[i]} {non_stop_words[i+1]} {non_stop_words[i+2]}"
                if trigram in raw_doc:
                    bonus += 4.5

        # Full phrase match
        full_phrase = " ".join(non_stop_words)
        if len(non_stop_words) >= 2 and full_phrase in raw_doc:
            bonus += 5.0

        return bonus

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[ScoredClause]:
        """
        Scores and ranks policy clauses for the given natural language query.
        
        Args:
            query: The user query string.
            top_k: Maximum number of ranked results to return.
            min_score: Optional override for minimum score threshold.
            
        Returns:
            List of ScoredClause instances sorted by score in descending order.
        """
        threshold = self.min_score if min_score is None else min_score

        query_tokens = tokenize(query, stem=True, remove_stopwords=True)
        if not query_tokens:
            return []

        # Count query term frequencies
        qtf: Dict[str, int] = {}
        for qt in query_tokens:
            qtf[qt] = qtf.get(qt, 0) + 1

        scores: List[Tuple[int, float]] = []

        for idx, clause in enumerate(self.clauses):
            doc_terms = self.doc_tokens[idx]
            doc_len = self.doc_lengths[idx]

            # Calculate term frequencies for this doc
            tf_map: Dict[str, int] = {}
            for t in doc_terms:
                tf_map[t] = tf_map.get(t, 0) + 1

            bm25_score = 0.0
            for term, q_count in qtf.items():
                if term not in tf_map:
                    continue

                tf = tf_map[term]
                idf = self.idf.get(term, 1.0)

                # BM25 core equation
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf * (num / denom)
                bm25_score += term_score

            # Add exact phrase bonus
            phrase_bonus = self._compute_exact_phrase_bonus(query, idx)
            total_score = bm25_score + phrase_bonus

            # Check for direct clause ID reference in query (e.g. "§4.3.2" or "4.3.2")
            if clause.clause_id in query or clause.citation in query:
                total_score += 20.0

            if total_score >= threshold:
                scores.append((idx, total_score))

        # Sort strictly deterministic: descending by score, tie-break by clause_id
        scores.sort(key=lambda item: (-item[1], self.clauses[item[0]].clause_id))

        results = [
            ScoredClause(clause=self.clauses[idx], score=score)
            for idx, score in scores[:top_k]
        ]
        return results


if __name__ == "__main__":
    import sys
    # Ensure UTF-8 output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    retriever = PolicyRetriever()

    demo_queries = [
        "What is the deadline for reporting a change?",
        "How much is the earnings disregard?",
        "Who is considered a full-time student?",
        "What happens if an overpayment was caused by Department error?",
    ]

    print("=" * 70)
    print("Policy Retriever Demonstration")
    print("=" * 70)

    for q in demo_queries:
        print(f"\nQuestion:\n{q}\n")
        results = retriever.retrieve(q, top_k=3)
        print("Retrieved clauses:")
        if not results:
            print("  (No matching clauses found above threshold)")
        else:
            for rank, res in enumerate(results, 1):
                preview = res.clause_text.replace("\n", " ")
                if len(preview) > 110:
                    preview = preview[:107] + "..."
                print(f"{rank}. {res.citation} ({res.clause_title or 'Clause'})")
                print(f"   Score:   {res.score:.4f}")
                print(f"   Section: {res.parent_section}")
                print(f"   Part:    {res.parent_part}")
                print(f"   Preview: {preview}\n")
        print("-" * 70)
