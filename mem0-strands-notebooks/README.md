# Agent memory with mem0 and Strands on AWS Bedrock

Four notebooks that take you from "what is agent memory" to a production quality gate, and then to a governed **customer context vault** shared by many agents, on one running example: an assistant for business banking relationship managers that remembers a customer across every conversation, channel and source system.

Each notebook has two halves. Part 1 is the memory fundamentals (the original series). The second half extends the same ideas to a banking customer context vault, organised around its nine problem spaces: ingestion, entity resolution, fact conflicts, memory model and lifecycle, retrieval, the Context Pack, grounding, evaluation, and the memory-layer choice.

Everything runs against real models (Claude on Bedrock, Titan embeddings) with the open-source `mem0ai` library and the Strands Agents SDK. Nothing is mocked.

| # | Notebook | Level | What you build | What you learn |
|---|----------|-------|----------------|----------------|
| 01 | `01_memory_foundations.ipynb` | 100 to 300 | mem0 on Bedrock + FAISS; the vault's foundations (Part 5) | The three kinds of memory; how extraction and search work; why keeping memory true is your job. Then: a source adaptor with lineage, entity resolution and quarantine, a typed memory model (layers, slots, direct vs derived, verified vs unverified), conflicts without overwriting, views as of any date, customer / banker / agent / conversation scopes |
| 02 | `02_integration_patterns.ipynb` | 200 to 400 | Three ways to wire mem0 into a Strands agent; one vault for many agents (Part 2) | Tool vs hooks vs `MemoryManager`; bind identity at construction; inject ephemerally; save in the background. Then: a memory service with ABAC (Cedar semantics) and audit, hybrid retrieval with an entity hop, MCP-style tools, Desktop vs VRM views, a supervisor and sub-agent with identity propagation, cross-channel write-back, corrections, the Context Pack as a contract |
| 03 | `03_evaluating_memory.ipynb` | 300 to 400 | A four-layer evaluation with floor and ceiling baselines; the vault at four levels (Part 2) | Where memory fails; LLM-as-judge done carefully; diagnosing a wrong answer to a layer. Then: evidence sufficiency from lineage, a retrieval ablation, unsafe retrievals, Context Pack quality, leakage across channels and purposes, simulated customers, judge consistency |
| 04 | `04_production_quality.ipynb` | 400 to 500 | Fixes, failure-mode tests, a release gate; operating the vault (Part 2) | Success criteria in numbers; isolation, poisoning, sensitive data, stale facts, abstention; governance. Then: dead-letter queues and late data, lineage-based deletion with link repair, a review queue, retention, temporal roll-ups, re-extraction by policy version, sensitive data in every mem0 table, tracing an answer, a vault gate, complete erasure, AgentCore mapping |
| 05 | `05_graph_memory_cookbook.ipynb` | 400 to 500 | Twenty recipes from *Graph-based Agent Memory* (arXiv 2602.05665) on the vault | Schema-guided triples, mention vs event time, bi-temporal edges, summary trees, hyperedges, routed and multi-round retrieval, hybrid sources, consolidation, significance, inferred links, experience memory, active inquiry, graph quality, relational privacy, poisoning, write conflicts, benchmarks, stability; ends with Q&A |

Read them in order. Each notebook is a story that ends where the next one starts.

## Setup

Requirements: Python 3.12, AWS credentials with Bedrock access in `us-east-1` (or set `AWS_REGION`) to:

- `us.anthropic.claude-haiku-4-5-20251001-v1:0` (the agent and the memory extractor)
- `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (the evaluation judge)
- `amazon.titan-embed-text-v2:0` (embeddings)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm      # enables mem0's entity matching in search
python -m ipykernel install --user --name mem0-strands --display-name "Python (mem0-strands)"
```

The notebooks are saved with the `Python (mem0-strands)` kernel selected. In VS Code, open a notebook and pick that kernel from the kernel picker (top right) if it is not already active; in JupyterLab it is selected automatically. Run the notebooks from this directory so the `memlab` package and the `data/` and `results/` folders resolve.

Model ids can be overridden with `MEMLAB_AGENT_MODEL`, `MEMLAB_JUDGE_MODEL`, `MEMLAB_EMBED_MODEL`. All local state (FAISS indexes, SQLite history, session files) is written under `.mem0_data/`, which is ignored by git. Runtimes: about 6, 10, 25 and 30 minutes. Rough cost of running all four notebooks once is under ten US dollars; notebooks 03 and 04 make several hundred model calls each.

## Layout

```
memlab/                 helper package; every module is code a notebook teaches first
  config.py             Bedrock model ids, the mem0 configuration, build_memory() and build_model()
  store.py              Mem0Store: a Strands MemoryStore backed by mem0 (notebook 02)
  vault.py              the customer context vault: adaptor, resolution, typed extraction, conflict rules, views, scopes (notebook 01, Part 5)
  access.py             principals, policy, MemoryService, hybrid retrieval, tools, VaultStore, Context Pack (notebook 02, Part 2)
  evaluation.py         judges, retrieval metrics, systems under test, failure diagnosis, vault evaluation and simulation (notebook 03)
  graph.py              graph memory: dated edges that inherit the asserting record's policy, traversal, the relationship tool (notebook 02 §2.8)
  techniques.py         one utility per technique in the graph-memory cookbook (notebook 05), each documented with its source, a runnable
                        example (doctests) and measured latency; `tools/benchmark_techniques.py` re-measures and updates the docstrings
data/
  memory_eval_dataset.json   two customers, seven dated sessions, 30 gold facts, 24 questions, isolation probes
  vault_dataset.json         party directory, system-of-record facts, nine more sources, banker memory, playbooks, principals, 16 vault questions, simulations
results/                scorecards, gate reports and charts written by notebooks 03 and 04
docs/                   the public-facing reading guide (architecture, patterns, evaluation, the vault, a study path)
  memory_graph_pipeline.md   every utility in build order (write path, evolution, read path, governance), with its source paper, a one-line description, its latency, and a diagram
```

To use a technique on its own: `from memlab.techniques import ground_time, as_known, multi_round_retrieve, ...`; every function's docstring gives its source, an example and its measured p50/p95 latency, `latency_report()` shows what your own calls cost, and `.venv/bin/python -m doctest memlab/techniques.py` checks the examples. Re-measure with `AWS_REGION=us-east-1 .venv/bin/python tools/benchmark_techniques.py`.

To re-run the whole series: `AWS_REGION=us-east-1 .venv/bin/python tools/run_all.py`. It executes 01 and 02 in parallel, then 03, then 04, retrying a failed notebook up to three times (logs in `results/logs/`), then rebuilds the guide and refreshes the numbers quoted in `../AgentLearning.md` with `tools/update_study_plan.py`.

`docs/index.html` (light) and `../Agent_memory.html` (dark, the same page) are generated from the results of notebooks 03 and 04 by `.venv/bin/python docs/build_guide.py`; re-run it after re-running the notebooks so every number in the guide matches the latest run. It needs the venv because the guide's access-policy table is computed from the rules in `memlab/access.py`. Open either in a browser; each is a single self-contained file.

The code in `memlab/vault.py` and `memlab/access.py` is organised in `# %% [name]` sections; `tools/generate_nb01.py` and `tools/generate_nb02.py` copy those sections into the notebooks verbatim, so edit the module and regenerate rather than editing the notebook.

## Versions and things that changed recently

Pinned in `requirements.txt`: `mem0ai 2.0.20`, `strands-agents 1.55.0`. Both libraries changed materially in 2026 and many online tutorials are out of date. The notebooks call these out where they matter; the short list:

- mem0 2.x `search()` and `get_all()` take `filters={"user_id": ...}` and reject a top-level `user_id=`. The `mem0_memory` tool in `strands-agents-tools` still passes it top-level, so its `retrieve` and `list` actions fail against mem0 2.x; notebook 02 shows the twenty-line replacement.
- mem0 2.x `add()` is add-only: a single extraction call, `ADD` events only, hash de-duplication. The two-call ADD/UPDATE/DELETE design in the mem0 paper is gone from the open-source library, and so are `custom_fact_extraction_prompt` and `custom_update_memory_prompt` (use `custom_instructions`). Notebook 04 shows how to reconcile superseded facts yourself, with an audit trail.
- mem0 open source has no timestamp parameter; relative dates are resolved against today. Put the conversation date in the text when ingesting history (notebook 03).
- Strands 1.55 ships a native `MemoryManager` plugin with a pluggable `MemoryStore` protocol, ephemeral injection and background extraction. That is the recommended integration (notebook 02, pattern C). `Agent.structured_output()` is deprecated in favour of `agent(prompt, structured_output_model=Model)`.
- mem0 sends anonymous telemetry by default; `memlab/__init__.py` sets `MEM0_TELEMETRY=false` before import.
- Each vector store implements its own subset of mem0's filter operators. FAISS implements only exact values and plain lists; an operator such as `{"in": [...]}` or `{"gte": ...}` silently matches nothing (notebook 01, Part 4.3).
- mem0 keeps two SQLite tables beside the vector store: `history` (old and new text of every change, including deletes) and `messages` (a rolling window of the ten most recent raw messages per scope, for `infer=True`). Redaction and erasure must reach both (notebook 04, Part 2).
