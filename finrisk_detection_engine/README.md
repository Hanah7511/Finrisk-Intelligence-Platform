
# FinRisk Detection Engine

> Multi-layer AML detection pipeline: rule-based checks, statistical and unsupervised anomaly detection, supervised XGBoost, graph features and risk fusion.

![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Supervised-green)

Part of the [Financial Risk Intelligence Platform](../README.md)

---

## Overview

The `finrisk_detection_engine` is the detection part of the Financial Risk Intelligence Platform. It loads transaction data, applies several detection layers, fuses their scores into one risk score and severity, and will pass alerts to `finintel_sentinel_ai` for investigation once that part is built.

---

## Detection Pipeline
```
DATA INGESTION              CSV + SQL Server + REST API                         built
    ↓
DATA VALIDATION             Missing values, schema, duplicates, types, ranges   built
    ↓
PREPROCESSING               Cleaning, encoding, outliers, timestamps, split     built
    ↓
FEATURE ENGINEERING         Derived and customer-history features               built
    ↓
RULE-BASED AML ENGINE       Large amount, structuring, sanctioned country,      built
                            geo-risk, velocity
    ↓
STATISTICAL DETECTION       Double MAD                                          built
    ↓
UNSUPERVISED DETECTION      Isolation Forest + One-Class SVM ensemble           built
    ↓
SUPERVISED DETECTION        XGBoost with SHAP explanations                      built
    ↓
GRAPH DETECTION             Ring benchmark (connected components);              built
                            graph features + XGBoost on IBM AML
    ↓
RISK FUSION                 Weighted score -> LOW / MEDIUM / HIGH / CRITICAL;   built (self-test)
                            a sanctioned country is never below HIGH
    ↓
END-TO-END RUN              All layers in one pipeline on the SQL data          planned
    ↓
→ Alerts passed to FinIntel Sentinel AI for investigation                       planned
```

Each layer can be run on its own, e.g. `python -m src.rules.rule_engine` or `python -m src.risk_fusion.risk_fusion`.

---

## Status

| Component | Status |
|---|---|
| Data ingestion & validation | Complete |
| Preprocessing & feature engineering | Complete |
| Rule-based engine | Complete |
| Statistical detection (double MAD) | Complete |
| Unsupervised ML (Isolation Forest + One-Class SVM) | Complete |
| Supervised XGBoost | Complete (self-test needs a stratified split) |
| Graph ring detection + IBM benchmark | Complete |
| Risk fusion & severity | Complete (self-test on hand-made rows) |
| End-to-end pipeline run | Planned |
| Random Forest, GraphSAGE GNN | Planned |
| FastAPI scoring API, MLflow, Docker, tests | Planned |

---

*Part of [Financial Risk Intelligence Platform](https://github.com/Hanah7511/Finrisk-Intelligence-Platform)*
