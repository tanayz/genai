"""A Strands MemoryStore backed by mem0 (introduced in notebook 02, pattern C)."""

import asyncio
from typing import Any

from strands.memory import ExtractionConfig, InvocationTrigger, MemoryEntry


def to_mem0_messages(messages: list[dict]) -> list[dict]:
    """Flatten Strands messages ({"role", "content": [blocks]}) into mem0 messages ({"role", "content": str})."""
    flat = []
    for message in messages:
        text = " ".join(block["text"] for block in message["content"] if "text" in block).strip()
        if text:
            flat.append({"role": message["role"], "content": text})
    return flat


class Mem0Store:
    """Long-term memory for ONE customer, served by mem0.

    The tenant identity (`user_id`) is bound when the store is created, never chosen by the model.
    Implements the two methods the Strands MemoryManager needs:
      - search():       recall, used for injection and for the `search_memory` tool
      - add_messages(): save, mem0 does the fact extraction server-side
    """

    def __init__(
        self,
        memory,
        user_id: str,
        *,
        name: str = "customer_memory",
        description: str = "Long-term facts about this customer from all previous conversations.",
        max_search_results: int = 5,
        extraction: ExtractionConfig | bool | None = None,
        metadata: dict[str, Any] | None = None,
        search_threshold: float = 0.1,
        hide_superseded: bool = False,
    ) -> None:
        self.memory = memory
        self.user_id = user_id
        self.name = name
        self.description = description
        self.max_search_results = max_search_results
        self.writable = True
        # Default: save after every agent invocation. Use IntervalTrigger(turns=N) to save less often.
        self.extraction = extraction if extraction is not None else ExtractionConfig(trigger=[InvocationTrigger()])
        self.metadata = metadata or {}
        self.search_threshold = search_threshold
        # Drop memories a later fact has marked as outdated (notebook 04, reconciliation).
        self.hide_superseded = hide_superseded
        self.last_search: list[dict] = []
        self.searches: list[list[dict]] = []   # every search this store served, for evaluation
        self.last_add: dict | None = None

    async def search(self, query: str, options: dict | None = None) -> list[MemoryEntry]:
        top_k = (options or {}).get("max_search_results") or self.max_search_results
        result = await asyncio.to_thread(
            self.memory.search,
            query,
            filters={"user_id": self.user_id},
            top_k=top_k,
            threshold=self.search_threshold,
        )
        hits = result["results"]
        if self.hide_superseded:
            hits = [h for h in hits if not (h.get("metadata") or {}).get("superseded_by")]
        self.last_search = hits
        self.searches.append(hits)
        return [
            MemoryEntry(
                content=item["memory"],
                metadata={
                    "score": item["score"],
                    "id": item["id"],
                    # the date a fact was last confirmed, so the model can tell old from new
                    "as_of": (item.get("updated_at") or item.get("created_at") or "")[:10],
                    **(item.get("metadata") or {}),
                },
            )
            for item in hits
        ]

    async def add_messages(self, messages: list[dict], context: Any = None) -> dict:
        flat = to_mem0_messages(messages)
        if not flat:
            return {"results": []}
        result = await asyncio.to_thread(self.memory.add, flat, user_id=self.user_id, metadata=self.metadata)
        self.last_add = result
        return result
