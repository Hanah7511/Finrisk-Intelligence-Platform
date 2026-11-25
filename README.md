# Financial Risk Intelligence Platform

> AML detection and financial crime investigation platform: a multi-layer detection pipeline (rules, statistics, machine learning and graph analysis), with a multi-agent investigation system planned.

![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This platform detects financial crime in transaction data and, once the investigation side is built, will hand suspicious activity to AI agents for investigation. The two parts are connected through a shared contract layer.

---

## Architecture
```
finrisk-intelligence-platform/
├── finrisk_detection_engine/     ⚙ AML detection pipeline (in progress, most layers built)
└── finintel_sentinel_ai/         🤖 Multi-agent investigation system (planned, not built yet)
```
### ⚙ finrisk_detection_engine

Built:
- Data loading from SQL Server (SQLAlchemy + pyodbc), with validation, preprocessing and feature engineering.
- Rule-based detection: large amounts, structuring, sanctioned countries, geo-risk, velocity.
- Statistical anomaly detection: double MAD.
- Unsupervised anomaly detection: Isolation Forest + One-Class SVM ensemble.
- Supervised detection: XGBoost with SHAP explanations.
- Graph analysis: ring detection benchmark (connected components) and graph features + XGBoost on the IBM AML benchmark.
- Risk fusion: combines the layer scores into one risk score and a severity (LOW / MEDIUM / HIGH / CRITICAL).

Planned:
- Wiring all layers into one end-to-end pipeline run.
- Random Forest layer, GraphSAGE GNN for ring detection.
- FastAPI scoring API, MLflow experiment tracking, Docker deployment, automated tests.

### 🤖 finintel_sentinel_ai (planned)
- Investigation, Compliance and Summarization agents (LangGraph).
- RAG over FATF and FinCEN regulatory documents.
- Every agent step checked by [FinGuard](https://github.com/Hanah7511/Finguard-LLM-Guardrails-Gateway), the guardrails gateway built for it.

---

## Pipeline Flow

Built: SQL Ingest → Validate → Preprocess → Features → Rules → Statistical → Unsupervised ML → XGBoost → Graph features → Fusion → Severity

Planned: end-to-end run → Alert → FinIntel investigation → SAR report

---

## Benchmark result (IBM AML HI-Small)

Graph features + XGBoost, time-based split (train Sept 1–5, validation Sept 6, test Sept 7 onward), graph features from a trailing window only, threshold tuned on validation:

| model | precision | recall | F1 | AUC-PR | flagged |
|---|---|---|---|---|---|
| transaction features only | 0.074 | 0.192 | 0.107 | 0.040 | 6,796 |
| transaction + graph features | 0.420 | 0.282 | 0.337 | 0.304 | 1,773 |

Test set: 1,790,598 transactions, 2,642 laundering (base rate 0.1475%).

---

## Tech Stack

| Layer | Used now | Planned |
|---|---|---|
| ML | Python, Scikit-learn, XGBoost, SHAP, NetworkX | PyTorch, GraphSAGE |
| Data | SQL Server, Pandas, SQLAlchemy | — |
| Agentic AI | — | LangGraph, OpenAI API |
| Vector / RAG | — | FAISS or ChromaDB, Sentence Transformers |
| API & Infra | — | FastAPI, MLflow, Docker, GitHub Actions |

---

## Repo Structure
```
finrisk-intelligence-platform/
├── shared/
│   ├── schemas/
│   ├── configs/
│   ├── utils/
│   └── contracts/
├── finrisk_detection_engine/
│   ├── src/
│   ├── configs/
│   └── main.py
├── finintel_sentinel_ai/        (folders only, not built yet)
├── data/
└── README.md
```
---

## Status

| Component | Status |
|---|---|
| finrisk_detection_engine | In progress: detection layers and risk fusion built; end-to-end run, API and GNN planned |
| finintel_sentinel_ai | Planned: not built yet |

---

*Built by [Hana Al Haris](https://github.com/Hanah7511)*
