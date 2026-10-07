"""Model and storage configuration shared by all notebooks (introduced in notebook 01)."""

import os
import shutil
from pathlib import Path

from mem0 import Memory
from strands.models import BedrockModel

REGION = os.environ.get("AWS_REGION", "us-east-1")

# The agent and the memory extractor use a fast, cheap model.
AGENT_MODEL_ID = os.environ.get("MEMLAB_AGENT_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
# The evaluation judge uses a stronger model than the system it grades.
JUDGE_MODEL_ID = os.environ.get("MEMLAB_JUDGE_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
EMBED_MODEL_ID = os.environ.get("MEMLAB_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
EMBED_DIMS = int(os.environ.get("MEMLAB_EMBED_DIMS", "1024"))

# All local state (FAISS indexes, SQLite history, session files) lives here.
DATA_DIR = Path(os.environ.get("MEMLAB_DATA_DIR", Path(__file__).resolve().parent.parent / ".mem0_data"))


def mem0_config(collection: str, custom_instructions: str | None = None, llm_model_id: str = AGENT_MODEL_ID) -> dict:
    """Return a mem0 OSS config: Bedrock LLM + Bedrock embeddings + local FAISS index."""
    store_dir = DATA_DIR / "faiss" / collection
    return {
        "llm": {
            "provider": "aws_bedrock",
            "config": {
                "model": llm_model_id,
                "temperature": 0.1,
                "max_tokens": 2000,
                "aws_region": REGION,
            },
        },
        "embedder": {
            "provider": "aws_bedrock",
            "config": {
                "model": EMBED_MODEL_ID,
                "embedding_dims": EMBED_DIMS,
                "aws_region": REGION,
            },
        },
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": collection,
                "path": str(store_dir),
                "embedding_model_dims": EMBED_DIMS,
                # cosine gives a score in [0, 1] that is easy to reason about
                "distance_strategy": "cosine",
            },
        },
        "history_db_path": str(DATA_DIR / "history" / f"{collection}.db"),
        "custom_instructions": custom_instructions,
    }


def build_memory(collection: str, fresh: bool = False, **kwargs) -> Memory:
    """Create a mem0 Memory for a named collection. `fresh=True` wipes any previous data first."""
    if fresh:
        reset_collection(collection)
    (DATA_DIR / "history").mkdir(parents=True, exist_ok=True)
    return Memory.from_config(mem0_config(collection, **kwargs))


def reset_collection(collection: str) -> None:
    """Delete the FAISS index and history database for a collection."""
    shutil.rmtree(DATA_DIR / "faiss" / collection, ignore_errors=True)
    db = DATA_DIR / "history" / f"{collection}.db"
    if db.exists():
        db.unlink()


def build_model(model_id: str = AGENT_MODEL_ID, temperature: float = 0.2, **kwargs) -> BedrockModel:
    """Create a Strands Bedrock model handle."""
    return BedrockModel(model_id=model_id, region_name=REGION, temperature=temperature, **kwargs)
