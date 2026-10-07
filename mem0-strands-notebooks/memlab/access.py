"""Agents ask the vault: policy-enforced, audited, hybrid retrieval and the Context Pack contract (introduced in notebook 02).

Sections marked `# %% [name]` are shown verbatim in notebook 02 (tools/generate_nb02.py reads them from this file).

The principle, from the use-case analysis: agents do not search raw customer history. They ask one memory service
for *permitted* context, as a named principal, for a stated purpose. The service evaluates policy on every record
before ranking, logs every retrieval, and returns memories that carry their lineage, date, verification and basis.
"""

import asyncio
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.memory import ExtractionConfig, InvocationTrigger, MemoryEntry

from memlab.config import build_model
from memlab.store import to_mem0_messages
from memlab.vault import Interaction, ingest, render, vault_rows, view, conflicts

TODAY = "2026-09-27"


# %% [principals]
@dataclass(frozen=True)
class Principal:
    """Who is asking. For an agent acting for someone, `via` records the chain so the audit shows both."""
    id: str
    role: str                       # relationship_manager | contact_centre | customer_agent | complaints_officer | marketing | system
    channel: str
    purposes: tuple
    portfolio: tuple = ()
    active_customer: str | None = None
    via: tuple = ()

    def acting_through(self, agent_name: str) -> "Principal":
        """A sub-agent acts with the CALLER's identity and rights, never its own (no confused deputy)."""
        return replace(self, via=self.via + (agent_name,))


def principals_from(config: dict) -> dict[str, Principal]:
    return {pid: Principal(pid, c["role"], c["channel"], tuple(c["purposes"]), tuple(c.get("portfolio", ())), c.get("active_customer"))
            for pid, c in config.items()}


# %% [policies]
@dataclass(frozen=True)
class Rule:
    effect: Literal["permit", "forbid"]
    name: str
    applies: Callable[[Principal, str, str, dict | None, str], bool]   # (principal, action, customer_id, record, purpose)
    cedar: str


CUSTOMER_FACING = ("customer_agent",)
ACTIVE_CUSTOMER_ROLES = ("customer_agent", "contact_centre", "complaints_officer", "marketing")

RULES = [
    Rule("permit", "rm-reads-own-portfolio",
         lambda p, a, c, r, purpose: a == "read" and p.role == "relationship_manager" and c in p.portfolio,
         'permit(principal, action == Action::"read", resource) when { principal.role == "relationship_manager" && resource.customer in principal.portfolio };'),
    Rule("permit", "channel-reads-active-customer",
         lambda p, a, c, r, purpose: a == "read" and p.role in ACTIVE_CUSTOMER_ROLES and c == p.active_customer,
         'permit(principal, action == Action::"read", resource) when { resource.customer == principal.active_customer };'),
    Rule("permit", "owner-reads-own-banker-memory",
         lambda p, a, c, r, purpose: a == "read" and c == p.id,
         'permit(principal, action == Action::"read", resource) when { resource.scope == principal.id };'),
    Rule("permit", "channels-write-unverified-for-active-customer",
         lambda p, a, c, r, purpose: a == "write" and (c == p.active_customer or c in p.portfolio),
         'permit(principal, action == Action::"write", resource) when { resource.customer == principal.active_customer || resource.customer in principal.portfolio };'),
    Rule("forbid", "purpose-not-granted-to-principal",
         lambda p, a, c, r, purpose: purpose not in p.purposes,
         'forbid(principal, action, resource) unless { context.purpose in principal.purposes };'),
    Rule("forbid", "purpose-incompatible-with-record",
         lambda p, a, c, r, purpose: a == "read" and r is not None and purpose not in (r.get("purposes") or ()),
         'forbid(principal, action == Action::"read", resource) unless { context.purpose in resource.purposes };'),
    Rule("forbid", "rm-notes-only-for-relationship-managers",
         lambda p, a, c, r, purpose: r is not None and r.get("audience") == "rm_only" and p.role != "relationship_manager",
         'forbid(principal, action, resource) when { resource.audience == "rm_only" && principal.role != "relationship_manager" };'),
    Rule("forbid", "internal-records-never-to-customer-channels",
         lambda p, a, c, r, purpose: r is not None and r.get("audience") in ("internal", "rm_only", "banker_private") and p.role in CUSTOMER_FACING,
         'forbid(principal, action, resource) when { resource.audience != "customer" && principal.role == "customer_agent" };'),
    Rule("forbid", "banker-memory-only-for-its-owner",
         lambda p, a, c, r, purpose: r is not None and r.get("audience") == "banker_private" and r.get("owner") != p.id,
         'forbid(principal, action, resource) when { resource.audience == "banker_private" && resource.owner != principal.id };'),
    Rule("forbid", "credit-decisions-use-verified-facts-only",
         lambda p, a, c, r, purpose: a == "read" and purpose == "credit_decision" and r is not None and r.get("verification") != "verified",
         'forbid(principal, action == Action::"read", resource) when { context.purpose == "credit_decision" && resource.verification != "verified" };'),
    Rule("forbid", "only-systems-of-record-write-verified",
         lambda p, a, c, r, purpose: a == "write" and r is not None and r.get("verification") == "verified" and p.role != "system",
         'forbid(principal, action == Action::"write", resource) when { resource.verification == "verified" && principal.role != "system" };'),
]


@dataclass
class Decision:
    allowed: bool
    permits: list[str]
    forbids: list[str]


def authorize(principal: Principal, action: str, customer_id: str, record: dict | None = None, purpose: str = "servicing") -> Decision:
    """Cedar semantics: deny by default, any matching forbid wins, otherwise at least one permit must match."""
    permits = [r.name for r in RULES if r.effect == "permit" and r.applies(principal, action, customer_id, record, purpose)]
    forbids = [r.name for r in RULES if r.effect == "forbid" and r.applies(principal, action, customer_id, record, purpose)]
    return Decision(bool(permits) and not forbids, permits, forbids)


# %% [retrieval]
CORRECTION_WORDS = re.compile(r"\b(wrong|incorrect|mistake|error|correct\w*|got it wrong|fixed)\b", re.I)
HISTORY_WORDS = re.compile(r"\b(history|previous|before|earlier|changed|change|each time|walk me through|timeline|as of|used to|originally)\b", re.I)
LAYER_ROUTES = {
    "commitment": re.compile(r"\b(promis\w*|commit\w*|owe|follow[- ]?up|due|outstanding|overdue|open|pending|by when|still)\b", re.I),
    "preference": re.compile(r"\b(prefer\w*|contact\w*|call|phone|sms|email|text|reach|copied|copy)\b", re.I),
    "episodic": re.compile(r"\b(happen\w*|issue|fault|problem|complain\w*|dispute|history|when did|last time|same)\b", re.I),
    "semantic": re.compile(r"\b(who|what is|how many|directors?|owns?|entity|fleet|trucks|products?|holds?|accountant)\b", re.I),
}
TOKEN = re.compile(r"[a-z0-9$][a-z0-9$,.\-]*[a-z0-9]|[a-z0-9]", re.I)
STOP = set("the a an of to and or in on for is are was were be been with by at as it its this that what which who how did do does "
           "we our us i my me she her he his they their any has have had from about there should".split())


def tokens(text: str) -> list[str]:
    return [t.lower().strip(".,") for t in TOKEN.findall(text) if t.lower() not in STOP]


def route_layers(query: str) -> set[str]:
    """Cheap, deterministic query understanding: which memory types is this question about?"""
    return {layer for layer, pattern in LAYER_ROUTES.items() if pattern.search(query)}


def lexical_scores(query: str, rows: list[dict]) -> dict[str, float]:
    """BM25-style keyword score, normalised to [0, 1]. FAISS has no keyword index, so the service provides one; on
    OpenSearch this is the store's own BM25. Reference numbers and amounts are where it beats embeddings."""
    q = set(tokens(query))
    if not q or not rows:
        return {}
    docs = {r["id"]: tokens(r["text"]) for r in rows}
    df = Counter(t for d in docs.values() for t in set(d))
    n, avg = len(docs), sum(len(d) for d in docs.values()) / len(docs)
    raw = {}
    for rid, d in docs.items():
        tf = Counter(d)
        raw[rid] = sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
                       for t in q if t in tf)
    # saturate rather than divide by the best match: dividing makes the best match 1.0 even when it only shares the
    # customer's first name with the question, which defeats the relevance threshold
    return {rid: s / (s + 3.0) for rid, s in raw.items()}


@dataclass
class Hit:
    row: dict
    score: float
    why: dict = field(default_factory=dict)


# %% [service]
class MemoryService:
    """The one door to the vault. Every call names a principal and a purpose; every record is authorised before it can
    be ranked; every retrieval is audited. Agents never hold the mem0 client."""

    def __init__(self, memory, *, today: str = TODAY, min_score: float = 0.30, token_budget: int = 1200, directory: dict | None = None):
        self.memory, self.today, self.min_score, self.token_budget = memory, today, min_score, token_budget
        self.audit: list[dict] = []
        self.directory, self.graph = directory, None

    def enable_graph(self) -> None:
        """Build the graph over every customer in the directory (notebook 02, §2.8); rebuilt after every write."""
        from memlab.graph import GraphIndex
        self.graph = GraphIndex.build(self.memory, list(self.directory["customers"]))

    def _log(self, principal: Principal, action: str, customer_id: str, purpose: str, **details) -> None:
        self.audit.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "principal": principal.id,
                           "via": " > ".join(principal.via) or None, "role": principal.role, "action": action,
                           "customer": customer_id, "purpose": purpose, **details})

    def permitted_rows(self, principal: Principal, customer_id: str, purpose: str, *, as_of: str | None = None) -> tuple[list[dict], int]:
        rows = view(vault_rows(self.memory, customer_id), as_of=as_of, today=self.today)
        allowed = [r for r in rows if authorize(principal, "read", customer_id, r, purpose).allowed]
        return allowed, len(rows) - len(allowed)

    def search(self, principal: Principal, customer_id: str, query: str, *, purpose: str = "servicing", k: int = 8,
               as_of: str | None = None, hybrid: bool = True, route: bool = True, expand: bool = True, graph: bool = True) -> list[Hit]:
        if not authorize(principal, "read", customer_id, None, purpose).allowed:
            self._log(principal, "search", customer_id, purpose, query=query, returned=[], denied="customer not permitted")
            return []
        rows, denied = self.permitted_rows(principal, customer_id, purpose, as_of=as_of)
        by_id = {r["id"]: r for r in rows}
        # 1 semantic candidates (mem0 + Titan), restricted to rows this principal may see in this view
        semantic = {h["id"]: h["score"] for h in self.memory.search(query, filters={"user_id": customer_id}, top_k=60, threshold=0.0)["results"]
                    if h["id"] in by_id}
        lexical = lexical_scores(query, rows) if hybrid else {}
        layers = route_layers(query) if route else set()
        wants_history = bool(HISTORY_WORDS.search(query)) or as_of is not None
        hits = []
        for rid, row in by_id.items():
            s, lx = semantic.get(rid, 0.0), lexical.get(rid, 0.0)
            if s == 0.0 and lx == 0.0:
                continue
            why = {"semantic": round(s, 3), "lexical": round(lx, 3)}
            score = 0.65 * s + 0.35 * lx if hybrid else s
            if row.get("layer") in layers:
                score += 0.08; why["layer"] = row["layer"]
            if route and "commitment" in layers and row.get("state") in ("open", "overdue"):
                score += 0.08; why["open item"] = row["state"]
            if not row.get("current", True) and not wants_history:
                score -= 0.12; why["history"] = "replaced"
            if row.get("corrects") and CORRECTION_WORDS.search(query):
                score += 0.20; why["correction"] = True
            if row.get("verification") == "verified":
                score += 0.02
            hits.append(Hit(row, score, why))
        hits.sort(key=lambda h: h.score, reverse=True)
        chosen = [h for h in hits if h.score >= self.min_score][:k]
        if expand and graph and self.graph is not None:
            # graph traversal: permitted, valid-on-the-date edges, up to two hops, each record explained by its path
            from memlab.graph import graph_records
            linked = graph_records(self.graph, principal, query, [h.row for h in chosen], purpose=purpose, as_of=as_of)
            chosen += [Hit(row, 0.0, {"graph path": path}) for row, path in linked]
        elif expand and chosen:
            # one graph hop: pull in memories that share a named entity with the top hits or the question
            q = query.lower()
            anchors = {e.lower() for h in chosen[:3] for e in (h.row.get("entities") or [])}
            anchors |= {e.lower() for r in rows for e in (r.get("entities") or []) if len(e) > 3 and e.lower() in q}
            # an entity on a quarter of the customer's records (the customer's own name) links everything, so it links nothing
            counts = Counter(e.lower() for r in rows for e in set(r.get("entities") or []))
            anchors -= {e for e, n in counts.items() if n > 0.25 * len(rows)}
            have = {h.row["id"] for h in chosen}
            linked = [Hit(r, 0.0, {"linked by entity": sorted(anchors & {e.lower() for e in r.get("entities") or []})[:3]})
                      for r in rows if r["id"] not in have and r.get("current", True) and anchors & {e.lower() for e in r.get("entities") or []}]
            linked.sort(key=lambda h: (h.row.get("verification") == "verified", h.row.get("observed_on") or ""), reverse=True)
            chosen += linked[:max(2, k // 4)]
        # token budget: stop adding once the rendered context would exceed it
        budget, kept = self.token_budget, []
        for h in chosen:
            cost = len(render(h.row)) / 4
            if cost > budget:
                break
            budget -= cost; kept.append(h)
        self._log(principal, "search", customer_id, purpose, query=query, as_of=as_of, returned=[h.row["id"] for h in kept],
                  denied=denied, sources=sorted({h.row["source_ref"] for h in kept}))
        return kept

    def write_interaction(self, principal: Principal, customer_id: str, interaction: Interaction, customer: dict) -> dict:
        """Agents may write, but only as unverified, and only for a customer they are acting for."""
        probe = {"verification": "verified" if interaction.authoritative else "unverified"}
        decision = authorize(principal, "write", customer_id, probe, principal.purposes[0])
        if not decision.allowed:
            self._log(principal, "write", customer_id, principal.purposes[0], source=interaction.source_ref, denied=decision.forbids or "no permit")
            raise PermissionError(f"{principal.id} may not write to {customer_id}: {decision.forbids or 'no permit'}")
        log = ingest(self.memory, interaction, customer_id, customer)
        if self.graph is not None:
            self.enable_graph()
        self._log(principal, "write", customer_id, principal.purposes[0], source=interaction.source_ref,
                  written=[w["id"] for w in log["written"]])
        return log


# %% [tools]
BANK_AGENT_PROMPT = """You are the bank's customer-context assistant, answering a banker about the customer named below.
Use your tools. Answer briefly and specifically, with dates, amounts and references. Say whether a fact is customer-stated,
recorded in a bank system, or an interpretation, and never present a line marked HISTORY as current.
If the tools return nothing relevant, say "I have nothing on file about that" and do not guess. Today is {today}."""

CUSTOMER_AGENT_PROMPT = """You are the bank's Virtual RM, talking directly with the customer named below. Use your tools for
anything about their history with the bank. Be warm, brief and specific. Never ask the customer for something your tools
already show. Internal bank notes, assessments and staff opinions are never shared with customers: if asked about them,
say you can't share internal notes, and never claim that none exist. More generally, when your tools show nothing
about something the customer asks the bank has noted, say it isn't available to you in this channel, not that no such note exists. If the tools return nothing relevant, say you don't
have that on file and offer to connect them with their relationship manager. Today is {today}."""


WITHHELD_NOTICE = ("Policy notice: some of this customer's records are internal to the bank and are not shown in this channel. "
                   "Never tell the customer that no such notes or records exist; say you can't see or share them here.")


def vault_tools(service: MemoryService, principal: Principal, customer_id: str, purpose: str = "servicing") -> list:
    """MCP-style tools for one principal and one customer. Nothing the model passes can change who or what they read."""

    @tool
    def customer_context(query: str) -> str:
        """Search the customer's memory for facts relevant to a question. Each line shows its type, date, verification and source.

        Args:
            query: what you need to know, in plain words
        """
        hits = service.search(principal, customer_id, query, purpose=purpose)
        notice = ("\n" + WITHHELD_NOTICE) if principal.role in CUSTOMER_FACING else ""
        return ("\n".join(render(h.row) for h in hits) or "No permitted memory matches this question.") + notice

    @tool
    def recall_as_of(date: str, query: str) -> str:
        """What the bank knew about the customer on a past date: facts and preferences as they stood then.
        To list commitments that were open on a past date, use open_commitments with as_of instead.

        Args:
            date: YYYY-MM-DD
            query: what you need to know
        """
        hits = service.search(principal, customer_id, query, purpose=purpose, as_of=date)
        return "\n".join(render(h.row) for h in hits) or f"No permitted memory as of {date} matches this question."

    @tool
    def open_commitments(as_of: str = "") -> str:
        """The complete list (not a ranked search) of the bank's commitments and the customer's requests that are open or
        overdue. Use it for ANY question about open, outstanding, pending or overdue items, today or on a past date.

        Args:
            as_of: optional YYYY-MM-DD to see what was open on that date; defaults to today
        """
        rows, _ = service.permitted_rows(principal, customer_id, purpose, as_of=as_of or None)
        items = [r for r in rows if r.get("state") in ("open", "overdue")]
        service._log(principal, "open_commitments", customer_id, purpose, as_of=as_of or None, returned=[r["id"] for r in items])
        return "\n".join(render(r) for r in items) or "No open commitments or requests."

    @tool
    def related_parties() -> str:
        """People and entities connected to the customer (directors, related companies, accountant), with the source of each."""
        rows, _ = service.permitted_rows(principal, customer_id, purpose)
        linked = [r for r in rows if r.get("current", True) and (r.get("attribute") in ("directors", "related_entity", "accountant", "relationship_manager") or r.get("relations"))]
        service._log(principal, "related_parties", customer_id, purpose, returned=[r["id"] for r in linked])
        return "\n".join(render(r) for r in linked) or "No related parties on file."

    tools = [customer_context, recall_as_of, open_commitments, related_parties]
    if service.graph is not None:
        from memlab.graph import graph_tool
        tools.append(graph_tool(service, principal, customer_id, purpose))
    return tools


# %% [store]
class VaultStore:
    """A Strands MemoryStore over the service: injection goes through policy and audit like every other read, and
    extraction writes the conversation back as an unverified interaction from this channel."""

    def __init__(self, service: MemoryService, principal: Principal, customer_id: str, customer: dict, *,
                 purpose: str = "servicing", name: str = "customer_vault", max_search_results: int = 8,
                 writable: bool = False, conversation_id: str = "live", today: str = TODAY):
        self.service, self.principal, self.customer_id, self.customer = service, principal, customer_id, customer
        self.purpose, self.name, self.max_search_results, self.today = purpose, name, max_search_results, today
        self.description = "The customer's permitted memory: typed, dated, sourced facts, preferences and commitments."
        # writable stores save each turn in the background, as an unverified interaction from this channel
        self.writable = writable
        self.extraction = ExtractionConfig(trigger=[InvocationTrigger()]) if writable else False
        self.conversation_id = conversation_id
        self.searches: list[list[dict]] = []

    async def search(self, query: str, options: dict | None = None) -> list[MemoryEntry]:
        k = (options or {}).get("max_search_results") or self.max_search_results
        hits = await asyncio.to_thread(self.service.search, self.principal, self.customer_id, query, purpose=self.purpose, k=k)
        self.searches.append([h.row for h in hits])
        entries = [MemoryEntry(content=render(h.row), metadata={"id": h.row["id"], "score": round(h.score, 3), **h.why}) for h in hits]
        if self.principal.role in CUSTOMER_FACING:
            # the channel sees a partial vault; say so, so that "nothing here" is never read as "nothing exists"
            entries.append(MemoryEntry(content=WITHHELD_NOTICE, metadata={"id": "policy-notice"}))
        return entries

    async def add_messages(self, messages: list[dict], context: Any = None) -> dict:
        flat = to_mem0_messages(messages)
        if not flat:
            return {"written": []}
        speaker = {"user": "Customer", "assistant": "Virtual RM" if self.principal.channel == "vrm" else "Banker"}
        interaction = Interaction(self.principal.channel, f"{self.conversation_id}-{len(self.service.audit)}", self.today,
                                  self.principal.channel, "transcript", "\n".join(f"{speaker[m['role']]}: {m['content']}" for m in flat),
                                  {"customer_id": self.customer_id}, author=self.principal.id)
        return await asyncio.to_thread(self.service.write_interaction, self.principal, self.customer_id, interaction, self.customer)


# %% [pack]
Basis = Literal["verified", "confirmed", "stated", "interpretation"]


class PackLine(BaseModel):
    text: str
    sources: list[str] = Field(description="8-character memory ids this line is based on")
    basis: Basis = "stated"


class NarrativePart(BaseModel):
    """The only fields a model writes. Everything else in the pack is computed from typed memory."""
    why_now: str = Field(description="One sentence: what the customer most likely wants from this interaction")
    changed_since_last_contact: list[PackLine] = Field(description="What is new or different since the previous contact, newest first")
    relevant_history: list[PackLine] = Field(description="Past events that matter for the focus, oldest first")
    next_best_action: str = Field(description="One concrete action for the banker or agent, grounded in the facts")


class ContextPack(BaseModel):
    customer_id: str
    audience: str
    as_of: str
    who: PackLine
    open_commitments: list[PackLine]
    preferences: list[PackLine]
    conflicts_to_check: list[str]
    why_now: str
    changed_since_last_contact: list[PackLine]
    relevant_history: list[PackLine]
    next_best_action: str
    sources: dict[str, str] = Field(description="memory id -> source_ref, for every id cited")
    confidence: Literal["high", "medium", "low"]


def basis_of(row: dict) -> Basis:
    """Confidence semantics, deterministic: verified by a system; confirmed by more than one source; stated once; or an interpretation."""
    if row.get("verification") == "verified":
        return "verified"
    if row.get("kind") == "derived":
        return "interpretation"
    return "confirmed" if row.get("confirmed_by") else "stated"


def line(row: dict, text: str | None = None) -> PackLine:
    return PackLine(text=text or row["text"], sources=[row["id"][:8]], basis=basis_of(row))


def build_context_pack(service: MemoryService, principal: Principal, customer_id: str, focus: str, *,
                       purpose: str = "servicing", banker_preferences: list[str] = (), max_lines: int = 6) -> ContextPack:
    rows, _ = service.permitted_rows(principal, customer_id, purpose)
    service._log(principal, "context_pack", customer_id, purpose, focus=focus, returned=[r["id"] for r in rows])
    current = [r for r in rows if r.get("current", True)]
    # deterministic fields: identity, commitments, preferences and conflicts come from typed rows, never from a model
    identity = next((r for r in current if r.get("attribute") == "business_name" and r.get("verification") == "verified"), None) \
        or next((r for r in current if r.get("layer") == "semantic"), current[0])
    commitments = sorted([r for r in current if r.get("state") in ("overdue", "open") and r.get("layer") == "commitment"],
                         key=lambda r: (r["state"] != "overdue", r.get("due_date") or "9999"))
    commitment_lines = [line(r, f"{r['state'].upper()}{' (due ' + r['due_date'] + ')' if r.get('due_date') else ''}: {r['text']}") for r in commitments]
    preferences = [line(r) for r in sorted((r for r in current if r.get("layer") == "preference"), key=lambda r: not r.get("attribute"))][:3]
    # the narrative fields: a model, given only permitted rows, writes why-now, changes, history and next action
    relevant = {h.row["id"] for h in service.search(principal, customer_id, focus, purpose=purpose, k=10)}
    facts = "\n".join(("* " if r["id"] in relevant else "  ") + render(r) for r in rows[:60])
    instructions = ("You write the narrative fields of a Context Pack for a bank. Use only the memory lines given; cite their ids. "
                    "Lines marked HISTORY were replaced; never present them as current. Keep each list to at most "
                    f"{max_lines} items. Distinguish what the customer said from what the bank's systems record. Cite EVERY memory id a line relies on, "
                    "and keep each line to facts those memories state.")
    if banker_preferences:
        instructions += " The reader's own preferences for briefs: " + " ".join(banker_preferences)
    writer = Agent(model=build_model(temperature=0.0), callback_handler=None, system_prompt=instructions)
    narrative = writer(f"Customer {customer_id}, as of {service.today}. Focus: {focus}\n"
                       f"Memory lines (newest first; * = most relevant to the focus):\n{facts}",
                       structured_output_model=NarrativePart).structured_output
    by_short = {r["id"][:8]: r for r in rows}
    for pl in narrative.changed_since_last_contact + narrative.relevant_history:
        cited = [by_short[s[:8]] for s in pl.sources if s[:8] in by_short]
        pl.sources = [r["id"][:8] for r in cited]      # drop any id the model invented
        pl.basis = min((basis_of(r) for r in cited), key=["interpretation", "stated", "confirmed", "verified"].index, default="interpretation")
    all_lines = [line(identity)] + commitment_lines + preferences + narrative.changed_since_last_contact + narrative.relevant_history
    cited = {s for pl in all_lines for s in pl.sources}
    conflict_notes = conflicts(rows)
    weak = sum(pl.basis == "interpretation" or not pl.sources for pl in all_lines)
    confidence = "low" if weak > len(all_lines) // 3 else ("medium" if conflict_notes or weak else "high")
    return ContextPack(customer_id=customer_id, audience=principal.role, as_of=service.today, who=line(identity),
                       open_commitments=commitment_lines[:max_lines], preferences=preferences, conflicts_to_check=conflict_notes,
                       why_now=narrative.why_now, changed_since_last_contact=narrative.changed_since_last_contact[:max_lines],
                       relevant_history=narrative.relevant_history[:max_lines], next_best_action=narrative.next_best_action,
                       sources={s: by_short[s]["source_ref"] for s in sorted(cited) if s in by_short}, confidence=confidence)


def render_pack(pack: ContextPack) -> str:
    def block(title, lines):
        return [f"{title}:"] + [f"  - {pl.text} [{', '.join(pl.sources) or 'no source'} | {pl.basis}]" for pl in lines] if lines else [f"{title}: none"]
    out = [f"CONTEXT PACK for {pack.customer_id} | audience {pack.audience} | as of {pack.as_of} | confidence {pack.confidence}",
           f"who: {pack.who.text} [{pack.who.basis}]", f"why now: {pack.why_now}"]
    out += block("open commitments", pack.open_commitments) + block("preferences", pack.preferences)
    out += block("changed since last contact", pack.changed_since_last_contact) + block("relevant history", pack.relevant_history)
    out += ["conflicts to check:"] + [f"  - {c}" for c in pack.conflicts_to_check or ["none"]]
    out += [f"next best action: {pack.next_best_action}"]
    return "\n".join(out)
