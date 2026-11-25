# FinIntel Sentinel AI

> Multi-agent AML investigation system: takes an alert from the detection engine, investigates it with AI agents that look up the law and the money trail, and drafts a Suspicious Activity Report (SAR) for a human analyst to decide on.

![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

Part of the [Financial Risk Intelligence Platform](../README.md).

---

## Overview

FinIntel receives alerts from `finrisk_detection_engine` (a final risk score, a severity and the layer that drove it). Three AI agents investigate the alert, check it against the law, and draft a SAR. Every agent step is checked by [FinGuard](https://github.com/Hanah7511/Finguard-LLM-Guardrails-Gateway), the guardrails gateway built for this system. FinIntel only drafts: **a human analyst decides whether a SAR is filed**.

**Built so far:** the law retrieval layer (chunking, embeddings, ChromaDB, search with citations).
**Prototyped:** the three agents' workflow, as a scripted agent in the FinGuard repo (`scripts/fake_agent.py`): it runs sample alerts through all of FinGuard's checks over MCP.
**Next:** the LLM agents and the tools they use.

---

## Design: hybrid agentic RAG

An investigation needs three kinds of information. Each is fetched in the way that fits it:

| Information | Example question | How FinIntel gets it | Status |
|---|---|---|---|
| **The law** (text) | "When must a bank file a SAR?" | **Vector RAG**: search by meaning, answer with the citation | **Built** |
| **The money trail** (connections) | "Who did this account send money to? Is any of them linked to a sanctioned party?" | **Graph tool**: query the transaction graph directly (the same graph the detection engine builds) | Planned |
| **Customer facts** (records) | "How old is the account? What KYC level?" | **SQL tool**: query SQL Server directly | Planned |

**Why not Graph RAG over documents?** In AML, the relationships live in the transaction data, not in the law text. Querying that data directly gives exact links that can be audited, instead of links an LLM extracted from text. The law corpus is small and already structured by citation, so plain vector search is enough for it.

```
                        FinRisk alert (e.g. HIGH, top_contributor = rules)
                                       │
                                       ▼
   ┌────────────────────── FinIntel agents (planned: LangGraph) ──────────────────────┐
   │  Investigation agent ──► Compliance agent ──► Summarization agent                  │
   │   graph tool, SQL tool     vector RAG (law)      drafts the SAR                     │
   └────────────────────────────────────┬────────────────────────────────────────────┘
                                        │  every step checked by FinGuard (MCP):
                                        │  tool permission, data masking, sanctions,
                                        │  legal claims grounded, final check
                                        ▼
                         SAR DRAFT  →  human analyst decides and files
```

### What the SAR draft contains (planned)

| Section | Filled by | Example |
|---|---|---|
| Who | Investigation (SQL tool) | Customer and linked accounts, masked by FinGuard |
| What | Investigation | Six cash deposits of $9,500 |
| When | Investigation | Dates of the activity |
| Where | Investigation (graph tool) | Branches, counterparties, countries |
| Why suspicious | Compliance (vector RAG) | "Matches structuring, 31 CFR 1010.100(xx)", with citation |
| How | Summarization | The pattern, e.g. fan-out then cycle |
| Evidence | All | Transaction IDs, FinRisk score and top contributor |
| Recommendation | Summarization | File / close, for the analyst to decide |

The SAR rule for banks (31 CFR 1020.320) is itself in the law index.

---

## Law retrieval (built)

```
US federal AML law (public domain, in the repo)              data/knowledge_base/us_aml_regulations.json
+ UAE law and FATF Recommendations (local only, if present)  data/knowledge_base/aml_regulations_local.json
      ↓ chunk   ~1,000 characters, 200 overlap, cut at sentence ends, citation kept on every chunk
      ↓ embed   sentence-transformers/all-MiniLM-L6-v2 (local, 384 numbers per chunk, normalised)
      ↓ store   ChromaDB on disk (cosine similarity), telemetry off
      ↓ query   question → top 3 chunks with jurisdiction, citation, similarity and official source link
```

| Source | Jurisdiction | Sections | In the repo? |
|---|---|---|---|
| US federal law (31 U.S.C. 5324; 31 CFR 1010.100, .230, .311, .313, .314, .410; 1020.320): CTR, aggregation, structuring, beneficial ownership, record keeping, bank SAR rule | US | 34 | Yes: public domain |
| Federal Decree-Law No. 10 of 2025 (official English translation, UAE Legislation portal) and CBUAE Guidance on Transaction Monitoring and Sanctions Screening | UAE | 69 | No: kept local until the terms of use are confirmed |
| FATF Recommendations (R.1, 6, 10, 11, 12, 16, 19, 20, 24) | International | 9 | No: copyrighted, all rights reserved |

The local file is gitignored, and so is the index (`vector_db/`), because it stores the text too. Without the local file, the index is built from US law only. Each entry records its source URL and retrieval date.

```bash
cd finintel_sentinel_ai
python -m src.rag.build_index                       # 112 sections -> 293 chunks (34 -> 74 with US law only)
python -m src.rag.query "When must a bank file a suspicious activity report?"
python -m src.rag.query --in UAE "When must suspicious transactions be reported to the FIU?"
```

`--in US | UAE | International` limits a search to one jurisdiction.

Examples: the SAR question returns 31 CFR 1020.320(a), (b) and (e). "Customer splits cash deposits to avoid the 10000 dollar report" returns the aggregation rule, the legal definition of structuring and the CTR filing rule, without the word "structuring" appearing in the question. With `--in UAE`, the FIU question returns the CBUAE guidance on filing STRs and SARs with the UAE Financial Intelligence Unit. "Politically exposed persons enhanced due diligence" returns FATF Recommendation 12.

Limits: a small English corpus; no re-ranking; no evaluation set yet to measure retrieval quality; UAE and FATF texts only on a computer that has the local file.

---

## Tech Stack

| Layer | Used now | Planned |
|---|---|---|
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) | — |
| Vector store | ChromaDB | — |
| Agents | — | LangGraph, OpenAI API |
| Tools | — | Transaction graph (NetworkX), SQL Server |
| Guardrails | FinGuard (separate repo, used by the prototype) | FinGuard MCP tools on every agent step |
| API / infra | — | FastAPI, Docker |

---

## Folder Structure

```
finintel_sentinel_ai/
├── src/
│   ├── chunking/       # chunker.py: splits law sections, keeps citations      (built)
│   ├── embeddings/     # embedder.py: local MiniLM embeddings                   (built)
│   ├── vector_store/   # chroma_store.py: ChromaDB index and search             (built)
│   ├── rag/            # build_index.py, query.py                               (built)
│   └── agents/         # investigation, compliance, summarization               (planned)
├── vector_db/          # the ChromaDB index, rebuilt locally (not committed: it holds the law text)
└── README.md
```

---

## Status

| Component | Status |
|---|---|
| Law retrieval (chunk, embed, ChromaDB, query with citations) | Built |
| Agent workflow | Prototyped (scripted agent in the FinGuard repo) |
| Investigation, Compliance, Summarization agents (LLM) | Planned |
| Graph and SQL tools | Planned |
| SAR draft | Planned |
| UAE law and FATF texts | Built (local only, not in the repo) |
| Retrieval evaluation | Planned |
| FastAPI, Docker | Planned |

---

*Part of [Financial Risk Intelligence Platform](https://github.com/Hanah7511/Finrisk-Intelligence-Platform)*
