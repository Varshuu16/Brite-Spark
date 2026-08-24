"""
Deterministic Lexical Policy Retriever using BM25 with multi-field weighting,
Porter stemming, concept expansion, cross-reference graph propagation, proximity scoring,
and temporal policy versioning.
"""

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from .models import PolicyClause
    from .loader import load_policy, load_full_policy_corpus
    from .temporal import TemporalContext, TemporalStatus, extract_temporal_context
except ImportError:
    from models import PolicyClause
    from loader import load_policy, load_full_policy_corpus
    from temporal import TemporalContext, TemporalStatus, extract_temporal_context


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


CONCEPT_SYNONYMS: Dict[str, List[str]] = {
    "deadlin": ["time", "limit", "period", "day", "within", "date", "deadlin"],
    "timefram": ["time", "limit", "period", "day", "within", "date"],
    "due": ["time", "limit", "period", "day", "within", "date"],
    "oblig": ["must", "requir", "oblig"],
    "penalti": ["sanction", "reduc", "penalti"],
    "disput": ["appeal", "review", "panel"],
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

        if w in {"apply", "applies", "applicant", "applicants", "application", "applications", "applicable"}:
            return "applic"
        if w in {"eligible", "eligibility", "ineligible", "ineligibility"}:
            return "eligib"
        if w in {"reside", "residence", "resident", "residential", "residing"}:
            return "resid"

        if w.endswith("sses"):
            w = w[:-2]
        elif w.endswith("ies"):
            w = w[:-2]
        elif w.endswith("ss"):
            pass
        elif w.endswith("s"):
            w = w[:-1]

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

        if w.endswith("y") and self.contains_vowel(w[:-1]):
            w = w[:-1] + "i"

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

        if w.endswith("e"):
            stem = w[:-1]
            m = self.get_measure(stem)
            if m > 1 or (m == 1 and not self.cvc(stem)):
                w = stem

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

    @property
    def effective_date(self) -> Optional[str]:
        return self.clause.effective_date

    @property
    def amended_by(self) -> Optional[str]:
        return self.clause.amended_by

    @property
    def is_amendment(self) -> bool:
        return self.clause.is_amendment

    @property
    def is_transitional(self) -> bool:
        return self.clause.is_transitional

    def to_dict(self) -> Dict[str, Any]:
        data = self.clause.to_dict()
        data["score"] = round(self.score, 4)
        return data

    def __str__(self) -> str:
        title_info = f" ({self.clause_title})" if self.clause_title else ""
        amend_info = f" [{self.amended_by}]" if self.amended_by else ""
        return f"[{self.citation}{title_info}{amend_info} | Score: {self.score:.4f}] {self.clause_text[:100]}..."


class PolicyRetriever:
    """
    Deterministic Lexical BM25 retriever with multi-field weighting, concept expansion,
    cross-reference citation graph propagation, phrase proximity scoring, and temporal versioning.
    """

    def __init__(
        self,
        clauses: Optional[List[PolicyClause]] = None,
        policy_path: Union[str, Path] = "data/policy-manual.md",
        amendment_path: Optional[Union[str, Path]] = "data/Amendment No. 2026-01.md",
        k1: float = 1.2,
        b: float = 0.75,
        min_score: float = 0.5,
    ):
        """
        Initializes the retriever and indexes the supplied clauses.
        
        Args:
            clauses: Optional pre-loaded list of PolicyClause objects.
            policy_path: Path to load policy clauses from if clauses not provided.
            amendment_path: Path to amendment file to include in corpus if available.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            min_score: Default minimum relevance threshold for returning results.
        """
        if clauses is not None:
            self.clauses = clauses
        else:
            self.clauses = load_full_policy_corpus(policy_path, amendment_path)

        self.k1 = k1
        self.b = b
        self.min_score = min_score

        self.weights = {
            "title": 4.0,
            "section": 3.0,
            "part": 1.0,
            "text": 1.5,
        }

        self._build_index()

    def _build_index(self):
        """Builds inverted index, document token statistics, cross-reference map, and IDF tables."""
        self.doc_count = len(self.clauses)
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[float] = []
        self.df: Dict[str, int] = {}
        self.clause_index: Dict[str, List[int]] = defaultdict(list)
        self.section_to_clause_indices: Dict[str, List[int]] = defaultdict(list)
        self.cross_refs: Dict[int, List[str]] = {}

        self.raw_texts: List[str] = []

        total_length = 0.0

        for idx, clause in enumerate(self.clauses):
            self.clause_index[clause.clause_id].append(idx)
            sec_num = clause.hierarchy.get("section_number")
            if sec_num:
                self.section_to_clause_indices[sec_num].append(idx)

            refs = re.findall(r"§(\d+\.\d+(?:\.\d+)?)", clause.clause_text)
            self.cross_refs[idx] = refs

            title_tokens = tokenize(clause.clause_title or "", stem=True, remove_stopwords=True)
            section_tokens = tokenize(clause.parent_section or "", stem=True, remove_stopwords=True)
            part_tokens = tokenize(clause.parent_part or "", stem=True, remove_stopwords=True)
            text_tokens = tokenize(clause.clause_text, stem=True, remove_stopwords=True)

            combined_tokens = []
            combined_tokens.extend(title_tokens * int(self.weights["title"]))
            combined_tokens.extend(section_tokens * int(self.weights["section"]))
            combined_tokens.extend(part_tokens * int(self.weights["part"]))
            combined_tokens.extend(text_tokens * int(self.weights["text"]))

            doc_len = len(combined_tokens)
            self.doc_tokens.append(combined_tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            unique_terms = set(combined_tokens)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1

            full_text = f"{clause.parent_part or ''} {clause.parent_section or ''} {clause.clause_title or ''} {clause.clause_text}".lower()
            self.raw_texts.append(full_text)

        self.avg_doc_len = total_length / self.doc_count if self.doc_count > 0 else 1.0

        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            val = (self.doc_count - freq + 0.5) / (freq + 0.5)
            self.idf[term] = math.log(1.0 + max(val, 0.01)) + 1.0

    def _compute_phrase_and_proximity_bonus(self, query_raw: str, doc_idx: int) -> float:
        """
        Computes bonus score for exact phrases and sliding-window term co-occurrence.
        """
        raw_doc = self.raw_texts[doc_idx]
        bonus = 0.0

        words = re.findall(r"\b[a-zA-Z0-9_\$]+\b", query_raw.lower())
        non_stop_words = [w for w in words if w not in STOP_WORDS]

        if len(non_stop_words) >= 2:
            for i in range(len(non_stop_words) - 1):
                bigram = f"{non_stop_words[i]} {non_stop_words[i+1]}"
                if bigram in raw_doc:
                    bonus += 3.0

        if len(non_stop_words) >= 3:
            for i in range(len(non_stop_words) - 2):
                trigram = f"{non_stop_words[i]} {non_stop_words[i+1]} {non_stop_words[i+2]}"
                if trigram in raw_doc:
                    bonus += 4.5

        full_phrase = " ".join(non_stop_words)
        if len(non_stop_words) >= 2 and full_phrase in raw_doc:
            bonus += 5.0

        doc_words = re.findall(r"\b[a-zA-Z0-9_\$]+\b", raw_doc)
        if len(non_stop_words) >= 2 and len(doc_words) >= 2:
            stems = [STEMMER.stem(w) for w in non_stop_words]
            doc_stems = [STEMMER.stem(w) for w in doc_words]
            found_pos = [i for i, s in enumerate(doc_stems) if s in stems]
            if len(found_pos) >= 2:
                for a in range(len(found_pos) - 1):
                    dist = found_pos[a+1] - found_pos[a]
                    if 1 < dist <= 8 and doc_stems[found_pos[a]] != doc_stems[found_pos[a+1]]:
                        bonus += 2.0
                        break

        return bonus

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        temporal_context: Optional[TemporalContext] = None,
    ) -> List[ScoredClause]:
        """
        Scores, temporally filters, and ranks policy clauses for the given natural language query.
        
        Args:
            query: The user query string.
            top_k: Maximum number of ranked results to return.
            min_score: Optional override for minimum score threshold.
            temporal_context: Optional pre-computed TemporalContext.
            
        Returns:
            List of ScoredClause instances sorted by score in descending order.
        """
        threshold = self.min_score if min_score is None else min_score

        base_tokens = tokenize(query, stem=True, remove_stopwords=True)
        if not base_tokens:
            return []

        t_ctx = temporal_context or extract_temporal_context(query)

        expanded_qtf: Dict[str, float] = {}
        for token in base_tokens:
            expanded_qtf[token] = expanded_qtf.get(token, 0.0) + 1.0
            if token in CONCEPT_SYNONYMS:
                for syn in CONCEPT_SYNONYMS[token]:
                    expanded_qtf[syn] = expanded_qtf.get(syn, 0.0) + 0.6

        raw_scores: Dict[int, float] = {}

        for idx, clause in enumerate(self.clauses):
            doc_terms = self.doc_tokens[idx]
            doc_len = self.doc_lengths[idx]

            tf_map: Dict[str, int] = {}
            for t in doc_terms:
                tf_map[t] = tf_map.get(t, 0) + 1

            bm25_score = 0.0
            for term, q_weight in expanded_qtf.items():
                if term not in tf_map:
                    continue

                tf = tf_map[term]
                idf = self.idf.get(term, 1.0)

                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf * (num / denom) * q_weight
                bm25_score += term_score

            phrase_bonus = self._compute_phrase_and_proximity_bonus(query, idx)
            total_score = bm25_score + phrase_bonus

            if clause.clause_id in query or clause.citation in query:
                total_score += 20.0

            if total_score > 0.0:
                raw_scores[idx] = total_score

        final_scores: Dict[int, float] = dict(raw_scores)
        for idx, score in raw_scores.items():
            if score < 3.0:
                continue
            refs = self.cross_refs.get(idx, [])
            for ref in refs:
                if ref in self.clause_index:
                    for target_idx in self.clause_index[ref]:
                        transfer = score * 0.30
                        final_scores[target_idx] = final_scores.get(target_idx, 0.0) + transfer
                elif ref in self.section_to_clause_indices:
                    sec_indices = self.section_to_clause_indices[ref]
                    transfer = (score * 0.25) / max(len(sec_indices), 1)
                    for target_idx in sec_indices:
                        final_scores[target_idx] = final_scores.get(target_idx, 0.0) + transfer

        for idx, clause in enumerate(self.clauses):
            if idx not in final_scores:
                continue

            current_score = final_scores[idx]

            if t_ctx.status == TemporalStatus.PRE_AMENDMENT:
                if not clause.is_amendment:
                    final_scores[idx] = current_score * 1.35
                elif clause.is_transitional:
                    final_scores[idx] = current_score * 1.10
                else:
                    final_scores[idx] = current_score * 0.75

                if t_ctx.applicable_transitional_rule and clause.clause_id == f"Amendment 2026-01 §{t_ctx.applicable_transitional_rule}":
                    final_scores[idx] += 25.0

            elif t_ctx.status == TemporalStatus.POST_AMENDMENT:
                if clause.is_amendment and not clause.is_transitional:
                    final_scores[idx] = current_score * 1.45
                elif clause.is_transitional:
                    final_scores[idx] = current_score * 1.15

                if t_ctx.applicable_transitional_rule and clause.clause_id == f"Amendment 2026-01 §{t_ctx.applicable_transitional_rule}":
                    final_scores[idx] += 25.0

            elif t_ctx.status == TemporalStatus.SPANNING:
                if clause.clause_id == "Amendment 2026-01 §5.3" or clause.clause_id == "7.4.3":
                    final_scores[idx] = current_score + 25.0
            elif t_ctx.status == TemporalStatus.UNSPECIFIED:
                if clause.is_transitional:
                    final_scores[idx] = current_score * 1.10

        qualified: List[Tuple[int, float]] = [
            (idx, score) for idx, score in final_scores.items() if score >= threshold
        ]

        qualified.sort(key=lambda item: (-item[1], self.clauses[item[0]].clause_id))

        results = [
            ScoredClause(clause=self.clauses[idx], score=score)
            for idx, score in qualified[:top_k]
        ]
        return results


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    retriever = PolicyRetriever()

    demo_queries = [
        "What is the deadline for reporting a change in February 2026?",
        "What is the deadline for reporting a change in April 2026?",
        "What is the earnings disregard for a determination on 15 March 2026?",
        "What is the deadline for reporting a change of circumstances?",
    ]

    print("=" * 70)
    print("Policy Retriever with Temporal Versioning Demonstration")
    print("=" * 70)

    for q in demo_queries:
        ctx = extract_temporal_context(q)
        print(f"\nQuestion: {q}")
        print(f"Detected Temporal Status: {ctx.status.value} (Event: {ctx.event_type.value}, Rule: {ctx.applicable_transitional_rule})")
        results = retriever.retrieve(q, top_k=4)
        print("Retrieved clauses:")
        for rank, res in enumerate(results, 1):
            amend_tag = f" [{res.amended_by}]" if res.amended_by else " [Original]"
            print(f" {rank}. {res.citation}{amend_tag} (Score: {res.score:.4f}): {res.clause_text[:75]}...")
        print("-" * 70)
