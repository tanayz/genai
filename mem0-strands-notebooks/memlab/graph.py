"""Graph memory over the vault: typed, dated edges that inherit the policy of the record asserting them (notebook 02, §2.8).

Sections marked `# %% [name]` are shown verbatim in notebook 02 (tools/generate_nb02.py reads them from this file).

The vault's records already carry `relations` (subject, predicate, object), from the extractor and from systems of
record. This module turns them into a graph without a new store: nodes are canonical entity names, and every edge
keeps the id, customer, lineage, verification and validity window of the record it came from. Two rules make it safe
for a bank:
  1. An edge is usable only if the principal may read the record that asserts it (the same `authorize` as notebook 02).
     A graph can link customers; it must not become a way to read one customer's records from another's context.
  2. An edge is valid from the record's `observed_on` until the record is superseded, so traversal is as-of-aware:
     "who were the directors on 1 August?" follows the edges valid on that date.
Plain Python adjacency lists are enough at this size; on AWS the same model lives in Neptune (GraphRAG), with the
same two rules enforced in the service that queries it.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

from memlab.access import TODAY, Principal, authorize
from memlab.vault import vault_rows


# %% [nodes]
def canonical(name: str) -> str:
    """One node per real-world entity: 'Sharma Logistics Pty Ltd', 'sharma logistics' and 'Sharma Logistics.' are one node."""
    name = re.sub(r"\s+pty\.?\s+ltd\.?$", "", name.strip(), flags=re.I)
    name = re.sub(r"^the\s+", "", name, flags=re.I)
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


@dataclass(frozen=True)
class Edge:
    subject: str                  # canonical node names
    predicate: str
    object: str
    record_id: str                # the record that asserts it: its policy and lineage are the edge's
    customer_id: str
    source_ref: str
    verification: str
    valid_from: str
    valid_to: str | None          # the date the asserting record was superseded, or None while it is current

    def valid_on(self, date: str) -> bool:
        return self.valid_from <= date and (self.valid_to is None or self.valid_to > date)


# %% [index]
class GraphIndex:
    """An in-memory graph over every customer's records. It holds everything; traversal filters by policy and date."""

    def __init__(self, edges: list[Edge], records: dict[str, dict], labels: dict[str, str]):
        self.edges, self.records, self.labels = edges, records, labels
        self.adjacent = defaultdict(list)
        for e in edges:
            self.adjacent[e.subject].append(e)
            self.adjacent[e.object].append(e)

    @classmethod
    def build(cls, memory, customer_ids) -> "GraphIndex":
        edges, records, labels = [], {}, {}
        for customer_id in customer_ids:
            for r in vault_rows(memory, customer_id):
                records[r["id"]] = r | {"customer_id": customer_id}
                for rel in r.get("relations") or []:
                    s, o = canonical(rel["subject"]), canonical(rel["object"])
                    if not s or not o or s == o:
                        continue
                    labels.setdefault(s, re.sub(r"\s+Pty\.? Ltd\.?$", "", rel["subject"].strip()))
                    labels.setdefault(o, re.sub(r"\s+Pty\.? Ltd\.?$", "", rel["object"].strip()))
                    edges.append(Edge(s, rel["predicate"].strip().lower().replace(" ", "_"), o, r["id"], customer_id, r["source_ref"],
                                      r.get("verification", "unverified"), r.get("observed_on") or "", r.get("superseded_on")))
        return cls(edges, records, labels)

    def anchors(self, text: str) -> set[str]:
        """Nodes named in a piece of text (whole-word match on the canonical name)."""
        t = f" {canonical(text)} "
        return {n for n in self.adjacent if len(n) > 3 and f" {n} " in t}

    def usable(self, e: Edge, principal: Principal, purpose: str, date: str) -> bool:
        return e.valid_on(date) and authorize(principal, "read", e.customer_id, self.records[e.record_id], purpose).allowed

    def paths(self, principal: Principal, anchors: set[str], *, purpose: str = "servicing", as_of: str | None = None,
              hops: int = 2, limit: int = 40) -> list[tuple[Edge, ...]]:
        """Every simple path of up to `hops` edges from the anchors, over edges this principal may read, valid on the date."""
        date, found, frontier = as_of or TODAY, [], [((), a) for a in anchors]
        for _ in range(hops):
            nxt = []
            for path, node in frontier:
                seen = {node} | {x for e in path for x in (e.subject, e.object)}
                for e in self.adjacent.get(node, []):
                    other = e.object if e.subject == node else e.subject
                    if other in seen or e in path or not self.usable(e, principal, purpose, date):
                        continue
                    found.append(path + (e,)); nxt.append((path + (e,), other))
            frontier = nxt
        unique = list(dict.fromkeys(found))
        return sorted(unique, key=len)[:limit]

    def render(self, path: tuple[Edge, ...]) -> str:
        return "  ;  ".join(f"{self.labels.get(e.subject, e.subject)} --{e.predicate}--> {self.labels.get(e.object, e.object)} "
                            f"[{e.source_ref}, from {e.valid_from}{', until ' + e.valid_to if e.valid_to else ''}]" for e in path)


# %% [retrieve]
def graph_records(graph: GraphIndex, principal: Principal, query: str, seed_rows: list[dict], *, purpose: str = "servicing",
                  as_of: str | None = None, hops: int = 2, k: int = 6) -> list[tuple[dict, str]]:
    """Retrieve, traverse, enrich: anchor on entities named in the question and in the top hits, walk permitted and valid
    edges, and return the records that assert the edges on the best paths, each with the path that explains it.
    Paths are ranked by how many of the question's words their node and predicate names contain."""
    anchors = graph.anchors(query) | {canonical(e) for r in seed_rows[:3] for e in (r.get("entities") or []) if canonical(e) in graph.adjacent}
    words = set(re.findall(r"[a-z]{4,}", query.lower()))
    def overlap(path):
        names = " ".join(f"{e.subject} {e.predicate.replace('_', ' ')} {e.object}" for e in path)
        return sum(w in names for w in words) - 0.1 * len(path)
    out, seen = [], {r["id"] for r in seed_rows}
    for path in sorted(graph.paths(principal, anchors, purpose=purpose, as_of=as_of, hops=hops), key=overlap, reverse=True):
        for e in path:
            if e.record_id not in seen:
                seen.add(e.record_id); out.append((graph.records[e.record_id], graph.render(path)))
        if len(out) >= k:
            break
    return out[:k]


# %% [tool]
def graph_tool(service, principal: Principal, customer_id: str, purpose: str = "servicing"):
    """An MCP-style tool: the relationship paths around an entity (or the customer's own business), as the principal may see them."""
    from strands import tool
    home = canonical(service.directory["customers"][customer_id]["business"])

    @tool
    def relationship_paths(entity: str = "", as_of: str = "") -> str:
        """Relationships around a person, business or asset connected to this customer (directors, owners, related entities,
        premises, facilities), as paths with their sources and dates. Use it for any question about how people and entities
        are connected, including who held a role on a past date.

        Args:
            entity: a name to start from; empty means the customer's own business
            as_of: optional YYYY-MM-DD; defaults to today
        """
        graph = service.graph
        anchors = graph.anchors(entity) if entity else set()
        paths = graph.paths(principal, anchors or {home}, purpose=purpose, as_of=as_of or None, hops=2)
        service._log(principal, "relationship_paths", customer_id, purpose, entity=entity or home, as_of=as_of or None,
                     returned=sorted({e.record_id for p in paths for e in p}))
        return "\n".join(graph.render(p) for p in paths[:25]) or "No permitted relationships found."

    return relationship_paths
