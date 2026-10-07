"""Utility functions for the graph-memory techniques in notebook 05 (the cookbook), one per source, with latency.

Each function names the technique and the work it comes from, as described in the survey *Graph-based Agent Memory:
Taxonomy, Techniques, and Applications* (Yang et al., arXiv 2602.05665, 2026). Where a function departs from the
original idea for a regulated setting, the docstring says how. The primary papers were not all read in full: treat
the attributions as "the idea, as the survey describes it", not as reimplementations.

Latency
-------
Every public function is wrapped in ``@timed(kind)``:

* ``deterministic``  pure Python over data you pass in; microseconds to milliseconds
* ``embedding``      calls the embedding model (Titan v2 on Bedrock) once per text
* ``model``          calls a language model (Claude on Bedrock); seconds per call

Each call is recorded in ``LATENCY``; ``latency_report()`` returns p50, p95 and call counts for what you ran.
``tools/benchmark_techniques.py`` runs every function on the vault, writes ``results/technique_latency.json``, and
rewrites the ``Measured latency:`` line in each docstring below, so the numbers you read here are the latest run.
Model latency depends on the region, the model, the load and the size of the input; read the numbers as an order of
magnitude, and measure in your own setting.

Quick start::

    from memlab.techniques import ground_time, latency_report
    ground_time("last Friday", "2026-09-08")      # TimeSpan(start='2026-09-04', end='2026-09-04', granularity='day', ...)
    latency_report()                              # p50 / p95 per function, for this session
"""

import functools
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Literal, NamedTuple

from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------------------------ latency recording
LATENCY: dict[str, list[float]] = defaultdict(list)
KIND: dict[str, str] = {}


def timed(kind: Literal["deterministic", "embedding", "model"]):
    """Record the wall-clock duration of every call in LATENCY, keyed by function name."""
    def wrap(fn):
        KIND[fn.__name__] = kind

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                LATENCY[fn.__name__].append(time.perf_counter() - started)
        return inner
    return wrap


def latency_report() -> list[dict]:
    """p50, p95, max and call count per function for this session, slowest first.

    >>> _ = days_between("2026-06-10", "2026-08-03")
    >>> any(row["function"] == "days_between" for row in latency_report())
    True
    """
    rows = []
    for name, seconds in LATENCY.items():
        ordered = sorted(seconds)
        pick = lambda p: ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]
        rows.append({"function": name, "kind": KIND.get(name, "?"), "calls": len(ordered),
                     "p50_ms": round(pick(50) * 1000, 3), "p95_ms": round(pick(95) * 1000, 3), "max_ms": round(ordered[-1] * 1000, 3)})
    return sorted(rows, key=lambda r: r["p50_ms"], reverse=True)


def reset_latency() -> None:
    LATENCY.clear()


def _agent(system_prompt: str, temperature: float = 0.0):
    """A Strands agent on the extraction model (imported lazily, so deterministic utilities need no AWS credentials)."""
    from strands import Agent
    from memlab.config import build_model
    return Agent(model=build_model(temperature=temperature), callback_handler=None, system_prompt=system_prompt)


def _structured(agent, prompt: str, model_cls, attempts: int = 3):
    for attempt in range(attempts):
        try:
            return agent(prompt, structured_output_model=model_cls).structured_output
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(3 * (attempt + 1))


# ================================================================================================ EXTRACTION (§IV)
# ---- Schema-guided triples: structured information extraction (§IV) and dynamic schema learning (§X)
PREDICATES = {
    "director_of": "a person is a director of a company", "owns": "an entity owns an asset or premises",
    "operates_from": "a business operates from premises", "tenant_of": "a business rents premises",
    "related_entity_of": "two companies are related", "accountant_of": "an external accountant acts for a business",
    "relationship_manager_of": "a banker manages a customer", "holds": "an entity holds a bank product or facility",
    "secured_by": "a facility is secured by an asset", "guarantees": "a person guarantees a facility",
    "plans": "a business plans a site, purchase or change", "employs": "a business employs a person or role",
}


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


class _Triples(BaseModel):
    triples: list[Triple]


@timed("model")
def extract_triples(text: str, schema: dict[str, str] | None = PREDICATES) -> tuple[list[Triple], list[Triple]]:
    """Extract (subject, predicate, object) triples, constrained to a predicate vocabulary.

    Source: structured information extraction into triples (survey §IV); dynamic schema learning is an open
    challenge (§X). Governed variant: the model must use the vocabulary or say OTHER; OTHER triples come back as
    schema-gap candidates for a person to review, instead of the model inventing predicate names.

    Returns (triples, gaps). Pass ``schema=None`` for free extraction (useful once, to discover a vocabulary).

    Example::

        triples, gaps = extract_triples("Customer: please copy my accountant, Deepak Rao, on loan paperwork. "
                                        "He does the BAS for Sharma Logistics.")
        # triples -> [Triple(subject='Deepak Rao', predicate='accountant_of', object='Sharma Logistics')]
        # gaps    -> [Triple(subject='Deepak Rao', predicate='OTHER', object='loan paperwork')]

    Cost: 1 model call. Measured latency (2026-10-06, us-east-1, 3 calls): p50 2.5 s, p95 2.5 s
    """
    rule = ("Use ONLY these predicates; if a relationship fits none, use predicate OTHER: " + json.dumps(schema)) if schema else \
           "Use short snake_case predicates of your choice."
    agent = _agent("Extract relationships between named people, businesses, assets and bank products as triples. " + rule)
    triples = _structured(agent, text, _Triples).triples
    if schema is None:
        return triples, []
    return [t for t in triples if t.predicate in schema], [t for t in triples if t.predicate not in schema]


# ---- TReMu: decouple mention time from event time; ground relative expressions at write time (§V)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

# NSW public holidays and the NSW bank holiday for 2026, for business-day arithmetic. Illustrative: check the official
# NSW list (and the bank's own operating calendar) before relying on it, and pass your own calendar in production.
NSW_2026_HOLIDAYS = frozenset({"2026-01-01", "2026-01-26", "2026-04-03", "2026-04-04", "2026-04-05", "2026-04-06",
                               "2026-06-08", "2026-08-03", "2026-10-05", "2026-12-25", "2026-12-28"})


class TimeSpan(NamedTuple):
    start: str
    end: str
    granularity: Literal["day", "week", "month", "year"]
    expression: str
    mention_date: str
    note: str = ""


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _month_span(y: int, m: int) -> tuple[str, str]:
    last = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
    return date(y, m, 1).isoformat(), last.isoformat()


@timed("deterministic")
def ground_time(expression: str, mention_date: str, tense: Literal["past", "future"] = "past",
                holidays: frozenset[str] = NSW_2026_HOLIDAYS) -> TimeSpan | None:
    """Resolve a time expression against the date it was said, with an explicit granularity.

    Source: TReMu's time-aware memorization, which decouples the mention time (session timestamp) from the inferred
    event time (survey §V). Deterministic rules rather than a model, so the result is auditable and repeatable; an
    expression the rules cannot ground safely returns None, to be flagged for review rather than guessed.

    ``tense`` decides ambiguous weekday and day-of-month references: "on Wednesday" in "we signed on Wednesday"
    (past) is the most recent Wednesday before the mention; in "I'll call on Wednesday" (future), the next one.

    >>> ground_time("last Friday", "2026-09-08")[:3]                 # said on a Tuesday
    ('2026-09-04', '2026-09-04', 'day')
    >>> ground_time("on Wednesday", "2026-09-04")[:2]                # said on a Friday, past tense
    ('2026-09-02', '2026-09-02')
    >>> ground_time("next Monday", "2026-09-04", tense="future")[:2]
    ('2026-09-07', '2026-09-07')
    >>> ground_time("by Friday the 25th of September", "2026-09-04", tense="future")[:3]
    ('2026-09-25', '2026-09-25', 'day')
    >>> ground_time("in July", "2026-09-04")[:3]                     # a month, not 1 July
    ('2026-07-01', '2026-07-31', 'month')
    >>> ground_time("within 5 business days", "2026-07-02", tense="future")[:2]
    ('2026-07-02', '2026-07-09')
    >>> ground_time("by Friday the 26th of September", "2026-09-04", tense="future").note
    'weekday does not match the date: 2026-09-26 is a Saturday'
    >>> ground_time("sometime soon", "2026-09-04") is None
    True

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 400 calls): p50 7 µs, p95 7 µs
    """
    e, m = expression.lower().strip(), _d(mention_date)
    span = lambda a, b, g, note="": TimeSpan(a.isoformat() if isinstance(a, date) else a, b.isoformat() if isinstance(b, date) else b,
                                             g, expression, mention_date, note)
    if x := re.search(r"(\d{4}-\d{2}-\d{2})", e):
        return span(x[1], x[1], "day")
    if x := re.search(r"within (\d+) business days?", e):
        return span(m, add_business_days(mention_date, int(x[1]), holidays), "day")
    if x := re.search(r"within (\d+) days?", e):
        return span(m, m + timedelta(days=int(x[1])), "day")
    if x := re.search(r"(?:(" + "|".join(WEEKDAYS) + r")\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(" + "|".join(MONTHS) + r")(?:\s+(\d{4}))?", e):
        month = MONTHS.index(x[3]) + 1
        year = int(x[4]) if x[4] else m.year + (1 if tense == "future" and month < m.month else 0) - (1 if tense == "past" and month > m.month else 0)
        d = date(year, month, int(x[2]))
        note = f"weekday does not match the date: {d} is a {WEEKDAYS[d.weekday()].title()}" if x[1] and WEEKDAYS.index(x[1]) != d.weekday() else ""
        return span(d, d, "day", note)
    if x := re.search(r"\b(last|next|this)?\s*(?:on\s+)?(" + "|".join(WEEKDAYS) + r")\b", e):
        target, rel = WEEKDAYS.index(x[2]), x[1] or ("last" if tense == "past" else "next")
        delta = (m.weekday() - target) % 7 or 7 if rel == "last" else (target - m.weekday()) % 7 or 7
        d = m - timedelta(days=delta) if rel == "last" else m + timedelta(days=delta)
        return span(d, d, "day")
    if x := re.search(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b", e):
        day = int(x[1])
        d = date(m.year, m.month, day)
        if tense == "future" and d <= m:
            d = date(m.year + (m.month == 12), m.month % 12 + 1, day)
        if tense == "past" and d > m:
            d = date(m.year - (m.month == 1), (m.month - 2) % 12 + 1, day)
        return span(d, d, "day", "month inferred from the mention date")
    if x := re.search(r"\b(?:in|during|by the end of|end of)\s+(" + "|".join(MONTHS) + r")(?:\s+(\d{4}))?", e):
        month = MONTHS.index(x[1]) + 1
        year = int(x[2]) if x[2] else m.year + (1 if tense == "future" and month < m.month else 0) - (1 if tense == "past" and month > m.month else 0)
        a, b = _month_span(year, month)
        return span(b, b, "day") if "end of" in e else span(a, b, "month")
    if re.search(r"\bend of (?:the |this )?month\b", e):
        return span(_month_span(m.year, m.month)[1], _month_span(m.year, m.month)[1], "day")
    if re.search(r"\b(today|this morning|this afternoon)\b", e):
        return span(m, m, "day")
    if re.search(r"\byesterday\b", e):
        return span(m - timedelta(days=1), m - timedelta(days=1), "day")
    if re.search(r"\btomorrow\b", e):
        return span(m + timedelta(days=1), m + timedelta(days=1), "day")
    if re.search(r"\blast week\b", e):
        start = m - timedelta(days=m.weekday() + 7)
        return span(start, start + timedelta(days=6), "week")
    return None


# ---- TReMu: symbolic date arithmetic instead of arithmetic in prose (§V)
@timed("deterministic")
def days_between(start: str, end: str) -> int:
    """Calendar days from start to end (negative if end is earlier).

    Source: TReMu's neuro-symbolic step: compute dates in code, not in the model's prose (survey §V).

    >>> days_between("2026-06-10", "2026-08-03")      # fault reported to technician fix
    54

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 2148 calls): p50 <1 µs, p95 1 µs
    """
    return (_d(end) - _d(start)).days


@timed("deterministic")
def add_business_days(start: str, n: int, holidays: frozenset[str] = NSW_2026_HOLIDAYS) -> str:
    """The date n business days after start, skipping weekends and the given holidays.

    >>> add_business_days("2026-07-02", 5)            # 'within 5 business days' of Thursday 2 July
    '2026-07-09'
    >>> add_business_days("2026-07-31", 1)            # Monday 3 August 2026 is the NSW bank holiday
    '2026-08-04'

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 6 µs, p95 6 µs
    """
    d = _d(start)
    while n:
        d += timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in holidays:
            n -= 1
    return d.isoformat()


class DateOperation(BaseModel):
    operation: Literal["days_between", "business_days_between", "add_days", "add_business_days", "earlier_of"]
    a: str = Field(description="YYYY-MM-DD")
    b: str | None = Field(default=None, description="YYYY-MM-DD, for two-date operations")
    n: int | None = Field(default=None, description="number of days, for add operations")
    reasoning: str


@timed("model")
def answer_temporal(question: str, timeline: list[tuple[str, str]]) -> dict:
    """Answer a date question by having the model choose a date operation and its arguments, then computing it in code.

    Source: TReMu generates Python for date arithmetic (survey §V). Governed variant: the model emits a structured
    operation from a fixed set (no arbitrary code is executed), and the arithmetic is done by the functions above.

    ``timeline`` is a list of (date, event) pairs, e.g. from the vault's records.

    Example::

        answer_temporal("How many days from the first fault report to the technician's fix?",
                        [("2026-06-10", "fault M-2287 reported"), ("2026-08-03", "technician swapped the terminal")])
        # -> {'operation': 'days_between', 'a': '2026-06-10', 'b': '2026-08-03', 'result': 54, ...}

    Cost: 1 model call. Measured latency (2026-10-06, us-east-1, 3 calls): p50 1.6 s, p95 1.6 s
    """
    agent = _agent("Choose ONE date operation and its arguments that answers the question, using only dates from the timeline.")
    op = _structured(agent, f"Question: {question}\n\nTimeline:\n" + "\n".join(f"{d}  {e}" for d, e in timeline), DateOperation)
    compute = {
        "days_between": lambda: days_between(op.a, op.b),
        "business_days_between": lambda: sum(1 for i in range(1, days_between(op.a, op.b) + 1)
                                            if (_d(op.a) + timedelta(days=i)).weekday() < 5
                                            and (_d(op.a) + timedelta(days=i)).isoformat() not in NSW_2026_HOLIDAYS),
        "add_days": lambda: (_d(op.a) + timedelta(days=op.n or 0)).isoformat(),
        "add_business_days": lambda: add_business_days(op.a, op.n or 0),
        "earlier_of": lambda: min(op.a, op.b),
    }
    return op.model_dump() | {"result": compute[op.operation]()}


# ---- MemoTime-inspired write-time validation: impossible orderings are flagged as they are written
@timed("deterministic")
def check_temporal_order(record: dict, rows: list[dict] = ()) -> list[str]:
    """Temporal-order violations in one vault record (and against the records it links to).

    Source: MemoTime enforces temporal monotonicity so reasoning never uses an effect that occurred before its cause
    (survey §V, hierarchical temporal constraints). This is the write-time counterpart: catch impossible orderings when
    the record is written, and route them to review like a system-of-record conflict.

    >>> check_temporal_order({"layer": "commitment", "observed_on": "2026-09-04", "due_date": "2026-09-01"})
    ['due date 2026-09-01 is before the promise was made (2026-09-04)']
    >>> check_temporal_order({"observed_on": "2026-07-02", "status": "closed", "closed_on": "2026-06-30"})
    ['closed on 2026-06-30, before it was recorded (2026-07-02)']
    >>> check_temporal_order({"observed_on": "2026-07-02", "closed_by": "x"}, [{"id": "x", "observed_on": "2026-06-01"}])
    ['closed by a record dated 2026-06-01, before this item (2026-07-02)']
    >>> check_temporal_order({"observed_on": "2026-09-04", "due_date": "2026-09-25", "layer": "commitment"})
    []

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 5 µs, p95 5 µs
    """
    seen, by_id, issues = record.get("observed_on") or "", {r.get("id"): r for r in rows}, []
    if record.get("layer") == "commitment" and record.get("due_date") and seen and record["due_date"] < seen:
        issues.append(f"due date {record['due_date']} is before the promise was made ({seen})")
    if record.get("closed_on") and seen and record["closed_on"] < seen:
        issues.append(f"closed on {record['closed_on']}, before it was recorded ({seen})")
    if record.get("superseded_on") and seen and record["superseded_on"] < seen:
        issues.append(f"superseded on {record['superseded_on']}, before it was recorded ({seen})")
    for link in ("closed_by", "superseded_by"):
        other = by_id.get(record.get(link))
        if other and (other.get("observed_on") or "") < seen:
            issues.append(f"{link.replace('_', ' ')} a record dated {other['observed_on']}, before this item ({seen})")
    return issues


@timed("deterministic")
def monotonic(events: list[tuple[str, str]]) -> list[str]:
    """Steps in a causal chain that go backwards in time. Empty means the chain is in a possible order.

    Source: MemoTime's monotonicity constraint t1 <= t2 <= t3 along a reasoning chain (survey §V).

    >>> monotonic([("2026-07-02", "waiver promised"), ("2026-09-16", "credit posted"), ("2026-09-14", "complaint about missing credit")])
    ['complaint about missing credit (2026-09-14) comes after credit posted (2026-09-16) in the chain but is dated earlier']

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 1 µs, p95 1 µs
    """
    return [f"{b} ({tb}) comes after {a} ({ta}) in the chain but is dated earlier" for (ta, a), (tb, b) in zip(events, events[1:]) if tb < ta]


@timed("deterministic")
def co_valid(windows: list[tuple[str, str | None]]) -> tuple[str, str | None] | None:
    """The period in which every relationship in a path held at once, or None if there is none.

    For structural paths (A director_of B; B owns C) the edges must hold *simultaneously* for the path to mean
    anything; for event chains use ``monotonic``. Windows are (valid_from, valid_to or None for current).

    >>> co_valid([("2019-03-04", "2026-08-31"), ("2019-03-04", None)])    # Arjun's directorship; the ownership
    ('2019-03-04', '2026-08-31')
    >>> co_valid([("2019-03-04", "2026-08-31"), ("2026-09-01", None)]) is None
    True

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 1 µs, p95 1 µs
    """
    start = max(w[0] for w in windows)
    ends = [w[1] for w in windows if w[1]]
    end = min(ends) if ends else None
    return (start, end) if end is None or start < end else None


# ================================================================================================ STORAGE (§V)
# ---- Graphiti / Zep: bi-temporal facts (valid time and transaction time), invalidate rather than overwrite
@dataclass(frozen=True)
class BiTemporalFact:
    subject: str
    predicate: str
    object: str
    valid_from: str
    valid_to: str | None
    recorded_at: str                       # when the system learned it (transaction time)
    invalidated_at: str | None = None      # when the system learned it had ended


@timed("deterministic")
def as_known(facts: list[BiTemporalFact], valid_on: str, known_on: str) -> list[BiTemporalFact]:
    """Facts true on ``valid_on``, as the system knew them on ``known_on``.

    Source: Graphiti (Zep) keeps valid time and transaction time per fact and invalidates instead of overwriting
    (survey §V, temporal knowledge graphs). Answers audit questions: "what did our records show on that date?".

    >>> facts = [BiTemporalFact("Arjun Sharma", "director_of", "Sharma Logistics", "2019-03-04", "2026-08-31",
    ...                         recorded_at="2019-03-04", invalidated_at="2026-09-10")]
    >>> [f.subject for f in as_known(facts, valid_on="2026-09-05", known_on="2026-09-05")]   # resignation not yet keyed
    ['Arjun Sharma']
    >>> [f.subject for f in as_known(facts, valid_on="2026-09-05", known_on="2026-09-27")]
    []

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 3 µs, p95 3 µs
    """
    out = []
    for f in facts:
        if f.recorded_at > known_on or f.valid_from > valid_on:
            continue
        ended = f.valid_to if (f.invalidated_at and f.invalidated_at <= known_on) else None
        if ended is None or ended > valid_on:
            out.append(f)
    return out


# ---- MemTree: hierarchical summary tree with recursive summaries (§V)
class TreeTopic(BaseModel):
    name: str
    summary: str = Field(description="two sentences with dates")
    member_ids: list[str]


class SummaryTree(BaseModel):
    root_summary: str
    topics: list[TreeTopic]


@timed("model")
def build_summary_tree(records: list[dict], max_topics: int = 6) -> dict:
    """Group records into topic nodes with summaries and a root summary; report coverage and compression.

    Source: MemTree routes information into topic nodes and recursively updates ancestor summaries (survey §V).
    One level of topics here; call again on the topics for deeper trees. A summary is derived memory: give it the most
    restrictive audience of its members, and recompute it when a member changes.

    ``records`` need ``id`` and ``text``. Returns ``{"tree": SummaryTree, "coverage": float, "duplicates": int,
    "compression": float}``; coverage below 1.0 means the tree silently dropped records.

    Example::

        result = build_summary_tree(vault_rows(memory, "cust_priya"))
        result["tree"].topics[0].name    # e.g. 'Merchant terminal faults'
        result["coverage"]               # 1.0 when every record is placed

    Cost: 1 model call; grows with the number of records. Measured latency (2026-10-06, us-east-1, 3 calls over 56 records): p50 13.8 s, p95 13.8 s
    """
    agent = _agent(f"Organise a bank customer's history into 3 to {max_topics} topics. Every record id must appear in exactly one topic. Cite ids.")
    tree = _structured(agent, "\n".join(f"[{r['id'][:8]}] {r.get('observed_on', '')} {r['text']}" for r in records), SummaryTree)
    known = {r["id"][:8] for r in records}
    placed = [i[:8] for t in tree.topics for i in t.member_ids if i[:8] in known]
    summary_chars = len(tree.root_summary) + sum(len(t.summary) for t in tree.topics)
    return {"tree": tree, "coverage": len(set(placed)) / max(1, len(known)), "duplicates": len(placed) - len(set(placed)),
            "compression": summary_chars / max(1, sum(len(r["text"]) for r in records))}


# ---- HyperGraphRAG: n-ary relations as hyperedges (§V)
REFERENCE = re.compile(r"\b[A-Z]{1,3}-\d{3,6}\b")


def _canon(name: str) -> str:
    name = re.sub(r"\s+pty\.?\s+ltd\.?$", "", name.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


@timed("deterministic")
def hyperedges(records: list[dict], parties: tuple[str, ...] = ("the bank",)) -> list[dict]:
    """One hyperedge per commitment: every participant, reference and date of one promise, kept together.

    Source: HyperGraphRAG treats a fragment and all its entities as one hyperedge, so n-ary facts are not split into
    pairs (survey §V). Query with ``hyperedges_with``.

    >>> h = hyperedges([{"id": "a1b2c3d4e5", "layer": "commitment", "text": "The bank will send the application to Priya and Deepak Rao by 25 Sep (re W-5530).",
    ...                  "entities": ["Priya Sharma", "Deepak Rao"], "due_date": "2026-09-25", "state": "overdue"}])
    >>> sorted(h[0]["members"])
    ['deepak rao', 'priya sharma', 'the bank', 'w-5530']
    >>> sorted(hyperedges([{"id": "x", "layer": "commitment", "text": "Resolve C-9912.", "entities": ["C-9912"]}])[0]["members"])
    ['c-9912', 'the bank']

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 400 calls over 73 records): p50 86 µs, p95 110 µs
    """
    out = []
    for r in records:
        if r.get("layer") != "commitment":
            continue
        # a reference keeps its hyphen whether it arrives as an entity or in the text, so C-9912 is one member, not two
        members = set(parties) | {e.lower() if REFERENCE.fullmatch(e) else _canon(e) for e in (r.get("entities") or [])} \
                  | {m.lower() for m in REFERENCE.findall(r["text"])}
        out.append({"id": r["id"][:8], "members": members, "due": r.get("due_date"), "state": r.get("state"), "text": r["text"]})
    return out


@timed("deterministic")
def hyperedges_with(edges: list[dict], member: str) -> list[dict]:
    """Hyperedges that include a member (a name, or a reference such as 'C-9912').

    >>> hyperedges_with([{"members": {"deepak rao", "c-9912"}, "text": "x"}], "C-9912")[0]["text"]
    'x'

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 2 µs, p95 2 µs
    """
    m = member.lower() if REFERENCE.fullmatch(member) else _canon(member)
    return [h for h in edges if m in h["members"]]


# ================================================================================================ RETRIEVAL (§VI)
# ---- Temporal operator: parse the query's time constraint (Zep validity windows; LiCoMemory decay) (§VI)
@timed("deterministic")
def parse_time_query(query: str, today: str) -> dict:
    """The time constraint in a question: an as-of date, an event window, or a request for the latest.

    Source: the survey's temporal-based operator (§VI): parsing the question's time range and limiting retrieval to it is
    credited to AssoMem [51]; the as-of view applies per-fact validity windows as in Zep [18].

    >>> parse_time_query("As of 1 August 2026, how did Priya want to be contacted?", "2026-09-27")
    {'as_of': '2026-08-01'}
    >>> parse_time_query("What happened in July?", "2026-09-27")
    {'window': ('2026-07-01', '2026-07-31')}
    >>> parse_time_query("Anything in the last 2 weeks?", "2026-09-27")
    {'window': ('2026-09-13', '2026-09-27')}
    >>> parse_time_query("What is the latest on Newcastle?", "2026-09-27")
    {'recency': True}

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 2 µs, p95 2 µs
    """
    q = query.lower()
    if x := re.search(r"as of (\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})", q):
        return {"as_of": date(int(x[3]), MONTHS.index(x[2]) + 1, int(x[1])).isoformat()}
    if x := re.search(r"\b(?:in|during) (" + "|".join(MONTHS) + r")(?: (\d{4}))?", q):
        month = MONTHS.index(x[1]) + 1
        year = int(x[2]) if x[2] else _d(today).year - (1 if month > _d(today).month else 0)
        return {"window": _month_span(year, month)}
    if x := re.search(r"last (\d+) (day|week)s?", q):
        days = int(x[1]) * (7 if x[2] == "week" else 1)
        return {"window": ((_d(today) - timedelta(days=days)).isoformat(), today)}
    return {"recency": bool(re.search(r"\b(latest|recent|now|current|currently)\b", q))}


@timed("deterministic")
def recency_decay(score: float, observed_on: str, today: str, half_life_days: float = 30.0, protected: bool = False) -> float:
    """Multiply a relevance score by exponential recency decay; protected records (commitments, verified facts) are not decayed.

    Source: LiCoMemory [52] applies a decay function during ranking for experience memory (survey §VI).

    >>> round(recency_decay(0.8, "2026-08-28", "2026-09-27", half_life_days=30), 3)    # one half-life old
    0.4
    >>> recency_decay(0.8, "2026-01-01", "2026-09-27", protected=True)                 # an old promise is still owed
    0.8

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 1 µs, p95 1 µs
    """
    if protected:
        return score
    return score * math.exp(-max(0, days_between(observed_on, today)) * math.log(2) / half_life_days)


# ---- Retrieval pipeline: semantic anchoring -> structured expansion -> policy-controlled stopping (§VI)
RELATIONSHIP_QUERY = re.compile(r"\b(who|owns?|owner|director|related|connected|premises|guarant\w*|accountant|copied|parent|subsidiar\w*)\b", re.I)


@timed("deterministic")
def route_to_graph(query: str) -> bool:
    """Whether a question should be expanded through the graph (structured expansion) or answered by search alone.

    Source: the survey's retrieval pipeline of semantic anchoring, structured expansion and policy-controlled stopping
    (§VI). In notebook 05 recipe 6, routing kept the graph's gain on relationship questions and cut tokens elsewhere.

    >>> route_to_graph("Who owns the premises Sharma Logistics operates from?")
    True
    >>> route_to_graph("How does Priya want to be contacted?")
    False

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 <1 µs, p95 <1 µs
    """
    return bool(RELATIONSHIP_QUERY.search(query))


# ---- Multi-round retrieval with a sufficiency check (§VI enhancement strategies; MemoTime is listed under multi-round)
class _Plan(BaseModel):
    sub_questions: list[str] = Field(description="2 to 4 short retrieval queries that together cover the question")


class _Sufficient(BaseModel):
    reasoning: str
    sufficient: bool
    missing: list[str]


@timed("model")
def multi_round_retrieve(search: Callable[[str], list[dict]], question: str, rounds: int = 2,
                         render: Callable[[dict], str] = lambda r: r["text"]) -> dict:
    """Decompose, retrieve, check sufficiency, retrieve again for what is missing.

    Source: multi-round retrieval with sub-query decomposition and a sufficiency check (survey §VI). ``search`` is any
    function from a query to records with an ``id``; pass ``MemoryService.search`` wrapped for one principal, so every
    round is still authorised and audited.

    Returns ``{"evidence": [records], "rounds": n, "queries": [...]}``.

    Example::

        search = lambda q: [h.row for h in service.search(alex, "cust_priya", q)]
        out = multi_round_retrieve(search, "Which commitments are open, and which are no longer open?")
        len(out["evidence"]), out["rounds"]

    Cost: 1 model call to plan, plus 1 per round to check (2 to 3 calls), plus the searches. Measured latency (2026-10-06, us-east-1, 3 calls): p50 8.6 s, p95 9.8 s
    """
    planner = _agent("Break a banker's question into retrieval queries.")
    queries = _structured(planner, question, _Plan).sub_questions
    evidence, asked, used = {}, list(queries), 0
    for used in range(1, rounds + 1):
        for q in queries:
            for r in search(q):
                evidence[r["id"]] = r
        checker = _agent("Decide whether the evidence answers every part of the question.")
        verdict = _structured(checker, f"Question: {question}\n\nEvidence:\n" + "\n".join(render(r) for r in evidence.values()), _Sufficient)
        if verdict.sufficient or not verdict.missing:
            break
        queries = verdict.missing
        asked += queries
    return {"evidence": list(evidence.values()), "rounds": used, "queries": asked}


# ---- Hybrid-source retrieval with an authority rule (§VI)
@timed("deterministic")
def resolve_by_authority(memory_fact: dict | None, system_fact: dict | None,
                         kind: Literal["fact", "personal"]) -> dict:
    """Which source answers, and whether a conflict must be shown.

    Source: hybrid-source retrieval; "facts favour verifiable, authoritative sources; personal details favour internal
    records matched to the right user and time" (survey §VI). A conflict is never resolved silently: both values are
    returned with a note, for the agent or the Context Pack to present.

    >>> resolve_by_authority({"value": "16 trucks", "source": "customer, 2026-07-28"}, {"value": "14 trucks", "source": "crm, 2025-10-06"}, "fact")["answer"]
    '14 trucks'
    >>> resolve_by_authority({"value": "16 trucks", "source": "customer, 2026-07-28"}, {"value": "14 trucks", "source": "crm, 2025-10-06"}, "fact")["conflict"]
    'crm, 2025-10-06 says 14 trucks; customer, 2026-07-28 says 16 trucks; the system of record is authoritative for facts'
    >>> resolve_by_authority({"value": "phone after 4pm", "source": "customer"}, None, "personal")["answer"]
    'phone after 4pm'

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 <1 µs, p95 1 µs
    """
    preferred, other = (system_fact, memory_fact) if kind == "fact" else (memory_fact, system_fact)
    chosen = preferred or other
    conflict = None
    if memory_fact and system_fact and memory_fact["value"] != system_fact["value"]:
        conflict = (f"{system_fact['source']} says {system_fact['value']}; {memory_fact['source']} says {memory_fact['value']}; "
                    f"the {'system of record' if kind == 'fact' else 'customer'} is authoritative for {kind == 'fact' and 'facts' or 'personal details'}")
    return {"answer": chosen["value"] if chosen else None, "source": chosen["source"] if chosen else None, "conflict": conflict}


# ================================================================================================ EVOLUTION (§VII)
# ---- Consolidation: merge near-duplicates, gated by a model (Mem0, FLEX) (§VII)
def _cosine(a, b) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


@timed("embedding")
def propose_merges(records: list[dict], embed: Callable[[str], list[float]], threshold: float = 0.80, same_layer: bool = True,
                   workers: int = 6) -> list[tuple[float, dict, dict]]:
    """Pairs of records similar enough to be candidates for consolidation, most similar first.

    Source: consolidation merges similar memories into schema nodes (survey §VII). This proposes; ``gate_merge`` decides.
    ``embed`` is any text-to-vector function, e.g. ``memory.embedding_model.embed`` from mem0.

    Example::

        pairs = propose_merges(current_rows, lambda t: memory.embedding_model.embed(t, "search"))
        pairs[0][0]   # similarity of the closest pair, e.g. 0.89

    ``workers`` embeds in parallel threads; embedding calls are independent, so wall time falls roughly with the
    worker count until the endpoint throttles. Use ``workers=1`` to measure the sequential cost.

    Cost: one embedding call per record. Measured latency (2026-10-06, us-east-1, 2 calls over 62 records, 6 workers): p50 2.3 s, p95 2.7 s
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        vectors = list(pool.map(lambda r: embed(r["text"]), records))
    pairs = [(_cosine(vectors[i], vectors[j]), records[i], records[j]) for i in range(len(records)) for j in range(i + 1, len(records))
             if not same_layer or records[i].get("layer") == records[j].get("layer")]
    return sorted((p for p in pairs if p[0] >= threshold), key=lambda p: p[0], reverse=True)


class _Same(BaseModel):
    reasoning: str
    same_fact: bool = Field(description="True only if both state the same fact, with no detail in one that contradicts the other")


@timed("model")
def gate_merge(a: dict, b: dict) -> tuple[bool, str]:
    """Whether two similar records state the same fact (a model gate on a proposed merge).

    Source: Mem0 and FLEX gate consolidation with a model judging information gain (survey §VII). Rules applied first,
    without a model call: never merge across different due dates, values or customers.

    Example::

        gate_merge({"text": "Credit the waiver by 11 September."}, {"text": "Credit the waiver by 25 September."})
        # -> (False, 'different due dates')

    Two paths with very different costs: a pair rejected by the rules returns in microseconds; a pair that reaches
    the judge costs one model call. The measured figure below is for the judge path.

    Cost: 0 or 1 model call (the judge model). Measured latency (2026-10-06, us-east-1, 3 calls, judge path): p50 12.2 s, p95 12.6 s
    """
    for key, label in (("due_date", "due dates"), ("value", "values"), ("customer_id", "customers")):
        if a.get(key) and b.get(key) and a[key] != b[key]:
            return False, f"different {label}"
    dates = lambda t: set(re.findall(r"\b\d{1,2}(?:st|nd|rd|th)? (?:" + "|".join(m.title() for m in MONTHS) + r")\b|\b\d{4}-\d{2}-\d{2}\b", t))
    if dates(a["text"]) and dates(b["text"]) and dates(a["text"]) != dates(b["text"]):
        return False, "different due dates"
    from memlab.evaluation import judge
    verdict = judge(f"A: {a['text']}\nB: {b['text']}", _Same)
    return verdict.same_fact, verdict.reasoning


# ---- Reorganisation by significance: PageRank variants and decay (§VII)
@timed("deterministic")
def significance(records: list[dict], edges: list[tuple[str, str]] = (), today: str = "2026-09-27", half_life_days: float = 90.0,
                 damping: float = 0.85, iterations: int = 40) -> list[tuple[float, dict]]:
    """Records ranked by PageRank over records and the entities they mention, times recency decay; least significant first.

    Source: significance-based pruning with PageRank variants and decay (survey §VII). For a bank this ranks what a
    person reviews first; it never authorises deletion. Commitments, verified facts and open items should be excluded
    before acting on the list.

    >>> recs = [{"id": "r1", "text": "a", "entities": ["Deepak Rao", "Sharma Logistics"], "observed_on": "2026-09-04"},
    ...         {"id": "r2", "text": "b", "entities": [], "observed_on": "2026-06-10"}]
    >>> [r["id"] for _, r in significance(recs, today="2026-09-27")]       # an unlinked, older record ranks lowest
    ['r2', 'r1']

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 10 calls over 73 records): p50 2.9 ms, p95 3.0 ms
    """
    links = defaultdict(set)
    for r in records:
        links["rec:" + r["id"]]
        for e in r.get("entities") or []:
            links["rec:" + r["id"]].add("ent:" + _canon(e)); links["ent:" + _canon(e)].add("rec:" + r["id"])
    for s, o in edges:
        links["ent:" + _canon(s)].add("ent:" + _canon(o)); links["ent:" + _canon(o)].add("ent:" + _canon(s))
    nodes = list(links)
    score = {n: 1 / len(nodes) for n in nodes}
    for _ in range(iterations):
        score = {n: (1 - damping) / len(nodes) + damping * sum(score[m] / len(links[m]) for m in links[n] if links[m]) for n in nodes}
    ranked = [(score["rec:" + r["id"]] * math.exp(-max(0, days_between(r["observed_on"], today)) / half_life_days), r) for r in records]
    return sorted(ranked, key=lambda x: x[0])


# ---- Graph reasoning: latent link prediction by rules (§VII; ToG and RoG are cited for transitive inference)
@dataclass(frozen=True)
class InferredEdge:
    subject: str
    predicate: str
    object: str
    confidence: float
    because: tuple


INFERENCE_RULES = (("director_of", "owns", "has_interest_in", 0.7), ("director_of", "holds", "is_connected_to_facility", 0.6))


@timed("deterministic")
def infer_links(triples: list[tuple[str, str, str]], rules=INFERENCE_RULES) -> list[InferredEdge]:
    """Edges implied by two-step rules (A p1 B, B p2 C => A new C), each with its premises and a confidence.

    Source: graph reasoning inserts latent transitive links ("associative thinking") (survey §VII). Always label the
    result as inferred, and let only principals who may read every premise see it.

    >>> infer_links([("Priya Sharma", "director_of", "Sharma Property Holdings"), ("Sharma Property Holdings", "owns", "Parramatta depot")])[0].object
    'Parramatta depot'

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls over 16 edges): p50 36 µs, p95 36 µs
    """
    out = []
    for p1, p2, new, conf in rules:
        for a in triples:
            for b in triples:
                if a[1] == p1 and b[1] == p2 and _canon(a[2]) == _canon(b[0]):
                    out.append(InferredEdge(a[0], new, b[2], conf, (a, b)))
    return out


# ---- Experience memory: skills from successes, lessons from failures (ExpeL, Matrix) (§VII-B)
class Lesson(BaseModel):
    kind: Literal["skill", "lesson"]
    statement: str
    applies_when: str
    evidence_ids: list[str]


class _Lessons(BaseModel):
    lessons: list[Lesson]


@timed("model")
def propose_lessons(history: list[dict], max_lessons: int = 4) -> list[Lesson]:
    """Reusable skills (what worked) and lessons (what failed) from a service history, with evidence ids.

    Source: feedback-driven adaptation: ExpeL and Matrix crystallise successes into skills and analyse failures into
    lessons learned (survey §VII-B). Governed variant: these are proposals for a person to approve into the agent's
    procedural scope; lessons citing ids that do not exist are dropped.

    Example::

        for lesson in propose_lessons(episodic_rows):
            print(lesson.kind, lesson.statement)
        # lesson  Escalate a promised fee waiver that is overdue instead of re-promising it ...

    Cost: 1 model call. Measured latency (2026-10-06, us-east-1, 3 calls over 56 records): p50 7.9 s, p95 8.0 s
    """
    agent = _agent(f"From a customer's service history, propose at most {max_lessons} reusable skills or lessons for the bank's "
                   "servicing team. Generalise beyond this customer; cite the evidence ids.")
    lessons = _structured(agent, "\n".join(f"[{r['id'][:8]}] {r.get('observed_on', '')} {r['text']}" for r in history), _Lessons).lessons
    known = {r["id"][:8] for r in history}
    return [l.model_copy(update={"evidence_ids": [i for i in l.evidence_ids if i[:8] in known]}) for l in lessons
            if any(i[:8] in known for i in l.evidence_ids)]


# ---- Active inquiry: detect gaps and ask to fill them (ProMem) (§VII-B)
@timed("deterministic")
def find_gaps(records: list[dict], required: tuple[str, ...], today: str, stale_after_days: int = 30) -> list[dict]:
    """Missing, stale or conflicting attribute slots in a customer's current records.

    Source: active inquiry: ProMem self-reflects on missing nodes or ambiguous edges and generates queries to fill
    them (survey §VII-B). Deterministic detection; ``phrase_inquiry`` turns one gap into a question.

    >>> gaps = find_gaps([{"attribute": "fleet_size", "value": "16 trucks", "verification": "unverified", "observed_on": "2026-07-28",
    ...                    "current": True, "conflicts_with": "x"}], required=("fleet_size", "staff_count"), today="2026-09-27")
    >>> sorted(g["kind"] for g in gaps)
    ['conflict', 'missing', 'stale']

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 203 calls over 73 records): p50 6 µs, p95 7 µs
    """
    slots = {r["attribute"]: r for r in records if r.get("attribute") and r.get("current", True)}
    gaps = [{"kind": "missing", "slot": s} for s in required if s not in slots]
    for s, r in slots.items():
        if r.get("verification") == "unverified" and not r.get("confirmed_by") and days_between(r["observed_on"], today) > stale_after_days:
            gaps.append({"kind": "stale", "slot": s, "value": r.get("value"), "since": r["observed_on"]})
        if r.get("conflicts_with"):
            gaps.append({"kind": "conflict", "slot": s, "value": r.get("value")})
    return gaps


class Inquiry(BaseModel):
    question: str
    slot: str


@timed("model")
def phrase_inquiry(gaps: list[dict]) -> Inquiry:
    """One short, friendly question that fills the most useful gap; never about anything not listed, never for sensitive values.

    Example::

        phrase_inquiry([{"kind": "missing", "slot": "staff_count"}]).question
        # 'How many people do you currently have on your team?'

    Cost: 1 model call. Measured latency (2026-10-06, us-east-1, 3 calls): p50 1.9 s, p95 1.9 s
    """
    agent = _agent("Choose the single most useful gap and phrase one question for a business customer. Never ask for card or "
                   "account numbers. Do not ask about anything not listed as a gap.")
    return _structured(agent, "Gaps:\n" + json.dumps(gaps), Inquiry)


# ================================================================================================ CHALLENGES (§X)
# ---- The quality of the memory graph: structural, semantic, temporal and operational criteria (§X)
@timed("deterministic")
def graph_quality(edges: list[dict], entities_named: set[str] = frozenset(), functional: frozenset[str] = frozenset({"owns"}),
                  today: str = "2026-09-27") -> dict:
    """Intrinsic quality metrics of a memory graph, without a judge.

    Source: the survey's first open challenge: graph memory needs multidimensional quality criteria (structural,
    semantic, temporal, operational) and metrics for the graph itself are scarce (§X). The semantic dimension (is each
    edge supported by its record?) needs a judge; see notebook 05 recipe 15.

    ``edges`` are dicts with subject, predicate, object, valid_from and valid_to.

    >>> q = graph_quality([{"subject": "a", "predicate": "owns", "object": "x", "valid_from": "2020-01-01", "valid_to": None},
    ...                    {"subject": "b", "predicate": "owns", "object": "x", "valid_from": "2021-01-01", "valid_to": None}],
    ...                   entities_named={"a", "b", "x", "c"})
    >>> q["functional predicates violated now"], q["entities named but not in the graph"]
    ({'x': 2}, 1)

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls over 16 edges): p50 12 µs, p95 14 µs
    """
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x
    for e in edges:
        parent[find(e["subject"])] = find(e["object"])
    nodes = {n for e in edges for n in (e["subject"], e["object"])}
    current = [e for e in edges if e["valid_from"] <= today and (e["valid_to"] is None or e["valid_to"] > today)]
    owners = Counter(o for o, _ in {(e["object"], e["subject"]) for e in current if e["predicate"] in functional})
    return {"nodes": len(nodes), "edges": len(edges), "connected components": len({find(n) for n in nodes}),
            "entities named but not in the graph": len(set(entities_named) - nodes),
            "distinct predicates": len({e["predicate"] for e in edges}),
            "edges with a closed validity window": sum(e["valid_to"] is not None for e in edges),
            "functional predicates violated now": {o: n for o, n in owners.items() if n > 1}}


# ---- Memory integrity: quarantine unverified structure (§X, adversarial manipulation of memory)
GOVERNED_PREDICATES = frozenset({"director_of", "owns", "guarantees", "secured_by"})


@timed("deterministic")
def quarantine(edges: list[dict], owned_by: dict[str, str], governed: frozenset[str] = GOVERNED_PREDICATES) -> tuple[list[dict], list[dict]]:
    """Split edges into (usable, quarantined): unverified edges with a governed predicate, or reaching another customer's entity, wait for review.

    Source: adversaries can manipulate memory contents to inject malicious knowledge; defences include content
    validation and auditing (survey §X). ``owned_by`` maps canonical entity names to the customer that owns them.

    >>> usable, held = quarantine([{"subject": "marcus chen", "predicate": "director_of", "object": "sharma logistics",
    ...                             "verification": "unverified", "customer_id": "cust_priya"}],
    ...                           owned_by={"chens bakery": "cust_marcus", "sharma logistics": "cust_priya"})
    >>> len(usable), held[0]["predicate"]
    (0, 'director_of')

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls over 16 edges): p50 7 µs, p95 7 µs
    """
    usable, held = [], []
    for e in edges:
        other = any(owned_by.get(n, e["customer_id"]) != e["customer_id"] for n in (e["subject"], e["object"]))
        (held if e.get("verification") != "verified" and (e["predicate"] in governed or other) else usable).append(e)
    return usable, held


# ---- Memory coordination in multi-agent systems (§X)
CHANNEL_AUTHORITY = {"kyc": 3, "crm": 3, "pds": 3, "desk": 2, "contact_centre": 2, "complaints": 2, "vrm": 1}


@timed("deterministic")
def resolve_concurrent_writes(a: dict, b: dict) -> dict:
    """Decide between two writes to the same slot from different agents, independent of the order they arrived in.

    Source: inconsistent memory updates across agents lead to conflicting decisions; synchronisation and role-aware
    access are open problems (survey §X). Rule: later event time wins; at the same event time, a system of record
    outranks a person, who outranks an AI agent; at the same rank, keep both as a conflict. Order-independent by
    construction: ``resolve_concurrent_writes(a, b) == resolve_concurrent_writes(b, a)``.

    >>> desk = {"value": "email only", "observed_at": "2026-09-27T10:15", "channel": "desk"}
    >>> vrm = {"value": "phone after 4pm", "observed_at": "2026-09-27T10:15", "channel": "vrm"}
    >>> resolve_concurrent_writes(desk, vrm)["outcome"], resolve_concurrent_writes(vrm, desk)["value"]
    ('higher-authority channel wins', 'email only')

    Cost: no model call. Measured latency (2026-10-06, us-east-1, 200 calls): p50 <1 µs, p95 <1 µs
    """
    if a["observed_at"] != b["observed_at"]:
        later = max((a, b), key=lambda w: w["observed_at"])
        return {"outcome": "later event wins", "value": later["value"]}
    ra, rb = CHANNEL_AUTHORITY.get(a["channel"], 0), CHANNEL_AUTHORITY.get(b["channel"], 0)
    if ra != rb:
        return {"outcome": "higher-authority channel wins", "value": (a if ra > rb else b)["value"]}
    return {"outcome": "same time, same authority: keep both as a conflict", "value": None, "candidates": sorted([a["value"], b["value"]])}


if __name__ == "__main__":
    import doctest
    print(doctest.testmod(optionflags=doctest.ELLIPSIS))
