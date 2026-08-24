"""
Deterministic temporal analysis and date parsing module for policy amendments.
Extracts dates, identifies semantic roles (change date vs determination date vs claim period),
and classifies temporal validity according to Amendment No. 2026-01 transitional rules.
"""

from dataclasses import dataclass
import datetime
from enum import Enum
import re
from typing import Dict, List, Optional, Tuple


class TemporalStatus(Enum):
    PRE_AMENDMENT = "PRE_AMENDMENT"     
    POST_AMENDMENT = "POST_AMENDMENT"   
    SPANNING = "SPANNING"               
    UNSPECIFIED = "UNSPECIFIED"         


class QueryEventType(Enum):
    DETERMINATION = "DETERMINATION"               
    CHANGE_OF_CIRCUMSTANCES = "CHANGE_OF_CIRCUMSTANCES"  
    SPANNING_PERIOD = "SPANNING_PERIOD"           
    GENERAL = "GENERAL"                           


AMENDMENT_EFFECTIVE_DATE = datetime.date(2026, 3, 1)

MONTHS_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

DATE_REGEX = (
    r"(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:,?\s+\d{1,2}(?:st|nd|rd|th)?)?,?\s+\d{4}|\b\d{4}-\d{2}-\d{2}\b"
)


@dataclass
class TemporalContext:
    """
    Structured temporal classification of a user query with semantic date role attribution.
    """
    status: TemporalStatus
    event_type: QueryEventType
    detected_date_str: Optional[str] = None
    detected_date: Optional[datetime.date] = None
    controlling_date: Optional[datetime.date] = None
    change_date: Optional[datetime.date] = None
    determination_date: Optional[datetime.date] = None
    claim_period_date: Optional[datetime.date] = None
    span_start: Optional[datetime.date] = None
    span_end: Optional[datetime.date] = None
    applicable_transitional_rule: Optional[str] = None  
    explanation: str = ""


def parse_date_string(date_text: str) -> Optional[datetime.date]:
    """
    Deterministically parses common date formats into a datetime.date object.
    """
    if not date_text:
        return None
    clean = re.sub(r"(st|nd|rd|th)", "", date_text.lower().strip())
    
    # ISO Format: YYYY-MM-DD
    iso_match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", clean)
    if iso_match:
        return datetime.date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))

    dmy_match = re.match(r"^(\d{1,2})\s+([a-z]+)\s+(\d{4})$", clean)
    if dmy_match:
        day = int(dmy_match.group(1))
        month_name = dmy_match.group(2)
        year = int(dmy_match.group(3))
        if month_name in MONTHS_MAP:
            return datetime.date(year, MONTHS_MAP[month_name], day)

    mdy_match = re.match(r"^([a-z]+)\s+(\d{1,2}),?\s+(\d{4})$", clean)
    if mdy_match:
        month_name = mdy_match.group(1)
        day = int(mdy_match.group(2))
        year = int(mdy_match.group(3))
        if month_name in MONTHS_MAP:
            return datetime.date(year, MONTHS_MAP[month_name], day)

    my_match = re.match(r"^([a-z]+)\s+(\d{4})$", clean)
    if my_match:
        month_name = my_match.group(1)
        year = int(my_match.group(2))
        if month_name in MONTHS_MAP:
            return datetime.date(year, MONTHS_MAP[month_name], 1)

    return None


def extract_semantic_dates(query: str) -> Dict[str, Tuple[str, datetime.date]]:
    """
    Extracts dates paired with their semantic roles (change, determination, claim_period)
    rather than treating all dates homogeneously.
    """
    lower = query.lower()
    semantic_dates: Dict[str, Tuple[str, datetime.date]] = {}

    change_patterns = [
        rf"(?:change\s+(?:of\s+circumstances\s+)?(?:that\s+)?(?:occurred|happened|arose)|moved|change\s+was|change)\s+(?:on|in|at)?\s*({DATE_REGEX})",
        rf"({DATE_REGEX})\s+(?:change|move)",
    ]
    for pat in change_patterns:
        m = re.search(pat, lower)
        if m:
            d_str = m.group(1).strip()
            parsed = parse_date_string(d_str)
            if parsed:
                semantic_dates["change"] = (d_str, parsed)
                break

    det_patterns = [
        rf"(?:determination|decision|determined|decided)\s+(?:was\s+)?(?:made\s+)?(?:on|in|at)?\s*({DATE_REGEX})",
        rf"({DATE_REGEX})\s+(?:determination|decision)",
    ]
    for pat in det_patterns:
        m = re.search(pat, lower)
        if m:
            d_str = m.group(1).strip()
            parsed = parse_date_string(d_str)
            if parsed:
                semantic_dates["determination"] = (d_str, parsed)
                break

    claim_patterns = [
        rf"(?:claim\s+period|claim\s+was\s+for|claim\s+in|for\s+a)\s+({DATE_REGEX})\s+(?:claim|period)?",
    ]
    for pat in claim_patterns:
        m = re.search(pat, lower)
        if m:
            d_str = m.group(1).strip()
            parsed = parse_date_string(d_str)
            if parsed:
                semantic_dates["claim_period"] = (d_str, parsed)
                break

    return semantic_dates


def extract_temporal_context(query: str) -> TemporalContext:
    """
    Extracts dates, identifies semantic roles, and determines temporal status and transitional rule.
    """
    lower_query = query.lower()

    is_reporting_query = any(k in lower_query for k in [
        "deadline", "report", "reporting", "change of circumstance", "safe harbor", "10 days", "14 days"
    ])
    is_determination_query = any(k in lower_query for k in [
        "disregard", "threshold", "sanction", "reduction", "determination", "determined", "calculate"
    ])

    if is_reporting_query and not is_determination_query:
        event_type = QueryEventType.CHANGE_OF_CIRCUMSTANCES
    elif is_determination_query and not is_reporting_query:
        event_type = QueryEventType.DETERMINATION
    elif is_reporting_query and is_determination_query:
        if any(k in lower_query for k in ["deadline", "reporting", "report"]):
            event_type = QueryEventType.CHANGE_OF_CIRCUMSTANCES
        else:
            event_type = QueryEventType.DETERMINATION
    else:
        event_type = QueryEventType.GENERAL

    all_dates_in_query: List[Tuple[str, datetime.date]] = []
    for dm in re.findall(DATE_REGEX, lower_query):
        parsed = parse_date_string(dm)
        if parsed:
            all_dates_in_query.append((dm, parsed))

    is_spanning = any(k in lower_query for k in ["spanning", "from", "between"]) and any(w in lower_query for w in ["to", "until", "through", "and"])
    if is_spanning and len(all_dates_in_query) >= 2:
        d1 = min(d[1] for d in all_dates_in_query)
        d2 = max(d[1] for d in all_dates_in_query)
        if d1 < AMENDMENT_EFFECTIVE_DATE and d2 >= AMENDMENT_EFFECTIVE_DATE:
            return TemporalContext(
                status=TemporalStatus.SPANNING,
                event_type=QueryEventType.SPANNING_PERIOD,
                detected_date_str=f"{d1.isoformat()} to {d2.isoformat()}",
                controlling_date=d1,
                span_start=d1,
                span_end=d2,
                applicable_transitional_rule="5.3",
                explanation="The claim period spans across 1 March 2026; daily apportionment applies under Amendment §5.3 and §7.4.3."
            )

    semantic_dates = extract_semantic_dates(query)
    change_info = semantic_dates.get("change")
    det_info = semantic_dates.get("determination")
    claim_info = semantic_dates.get("claim_period")

    change_date = change_info[1] if change_info else None
    det_date = det_info[1] if det_info else None
    claim_date = claim_info[1] if claim_info else None

    controlling_date: Optional[datetime.date] = None
    controlling_str: Optional[str] = None
    rule: Optional[str] = None
    explanation: str = ""

    if event_type == QueryEventType.CHANGE_OF_CIRCUMSTANCES:
        rule = "5.2"
        if change_date:
            controlling_date = change_date
            controlling_str = change_info[0]
            if controlling_date < AMENDMENT_EFFECTIVE_DATE:
                status = TemporalStatus.PRE_AMENDMENT
                explanation = (
                    f"Change occurred on {controlling_str} (before 1 March 2026). "
                    f"Under Amendment §5.2, the pre-amendment 10-calendar-day reporting rule applies "
                    f"regardless of the determination date."
                )
            else:
                status = TemporalStatus.POST_AMENDMENT
                explanation = (
                    f"Change occurred on {controlling_str} (on or after 1 March 2026). "
                    f"Under Amendment §5.2, the amended 14-calendar-day reporting rule applies."
                )
            return TemporalContext(
                status=status,
                event_type=event_type,
                detected_date_str=controlling_str,
                detected_date=controlling_date,
                controlling_date=controlling_date,
                change_date=change_date,
                determination_date=det_date,
                claim_period_date=claim_date,
                applicable_transitional_rule=rule,
                explanation=explanation,
            )
        elif det_date:
            controlling_date = det_date
            controlling_str = det_info[0]
            status = TemporalStatus.PRE_AMENDMENT if controlling_date < AMENDMENT_EFFECTIVE_DATE else TemporalStatus.POST_AMENDMENT
            return TemporalContext(
                status=status,
                event_type=event_type,
                detected_date_str=controlling_str,
                detected_date=controlling_date,
                controlling_date=controlling_date,
                determination_date=det_date,
                applicable_transitional_rule=rule,
                explanation=f"Determination date {controlling_str} identified; §5.2 governs reporting rules.",
            )

    elif event_type == QueryEventType.DETERMINATION or is_determination_query:
        rule = "5.1"
        if det_date:
            controlling_date = det_date
            controlling_str = det_info[0]
            if controlling_date >= AMENDMENT_EFFECTIVE_DATE:
                status = TemporalStatus.POST_AMENDMENT
                explanation = (
                    f"Determination made on {controlling_str} (on or after 1 March 2026). "
                    f"Under Amendment §5.1, amended figures ($175 disregard / 15% sanction) apply, "
                    f"even for prior claim periods."
                )
            else:
                status = TemporalStatus.PRE_AMENDMENT
                explanation = (
                    f"Determination made on {controlling_str} (before 1 March 2026). "
                    f"Under Amendment §5.1, pre-amendment figures ($120 disregard / 20% sanction) apply."
                )
            return TemporalContext(
                status=status,
                event_type=QueryEventType.DETERMINATION,
                detected_date_str=controlling_str,
                detected_date=controlling_date,
                controlling_date=controlling_date,
                change_date=change_date,
                determination_date=det_date,
                claim_period_date=claim_date,
                applicable_transitional_rule=rule,
                explanation=explanation,
            )
        elif claim_date:
            controlling_date = claim_date
            controlling_str = claim_info[0]
            status = TemporalStatus.PRE_AMENDMENT if controlling_date < AMENDMENT_EFFECTIVE_DATE else TemporalStatus.POST_AMENDMENT
            return TemporalContext(
                status=status,
                event_type=QueryEventType.DETERMINATION,
                detected_date_str=controlling_str,
                detected_date=controlling_date,
                controlling_date=controlling_date,
                claim_period_date=claim_date,
                applicable_transitional_rule=rule,
                explanation=f"Claim period {controlling_str} identified under §5.1.",
            )

    if all_dates_in_query:
        first_str, first_date = all_dates_in_query[0]
        status = TemporalStatus.PRE_AMENDMENT if first_date < AMENDMENT_EFFECTIVE_DATE else TemporalStatus.POST_AMENDMENT
        rule = "5.2" if event_type == QueryEventType.CHANGE_OF_CIRCUMSTANCES else "5.1"
        return TemporalContext(
            status=status,
            event_type=event_type,
            detected_date_str=first_str,
            detected_date=first_date,
            controlling_date=first_date,
            applicable_transitional_rule=rule,
            explanation=f"Date {first_str} identified; governed by Amendment §{rule}.",
        )

    return TemporalContext(
        status=TemporalStatus.UNSPECIFIED,
        event_type=event_type,
        detected_date_str=None,
        detected_date=None,
        controlling_date=None,
        applicable_transitional_rule=None,
        explanation="No specific date was provided. Both pre-amendment and post-amendment rules must be presented with transitional rules."
    )
