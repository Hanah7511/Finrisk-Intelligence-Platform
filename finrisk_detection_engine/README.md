
# FinRisk Detection Engine

> Multi-layer AML detection pipeline combining rule-based engines, statistical anomaly detection, ensemble ML, Graph Neural Networks, and supervised fraud classification for real-time financial crime scoring.

![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble%20ML-green)

Part of the [Financial Risk Intelligence Platform](../README.md)

---

## Overview

The `finrisk_detection_engine` is the detection backbone of the Financial Risk Intelligence Platform. It ingests raw transaction data, applies multiple detection layers, fuses risk signals, and generates alerts that are passed to `finintel_sentinel_ai` for autonomous investigation.

---

## Detection Pipeline
```
MULTI-SOURCE DATA LAYER
(Banking Transactions + Customer Data + Device Data + KYC Records + Regulatory Documents)
    ↓
DATA INGESTION LAYER
(CSV + SQL Server + APIs + Kafka Streaming)
    ↓
DATA VALIDATION LAYER
(Missing Values + Schema Validation + Duplicate Checks + Type Validation)
    ↓
DATA PREPROCESSING LAYER
(Cleaning + Encoding + Scaling + Timestamp Processing + Feature Normalization)
    ↓
SQL ANALYTICS & HISTORICAL FEATURE LAYER
(Transaction Aggregation + Behavioral History + Velocity Metrics + Device Intelligence)
    ↓
FEATURE ENGINEERING LAYER
(SQL Behavioral Features + Python ML Features + Velocity Features + Risk Indicators)
    ↓
RULE-BASED AML ENGINE
(Structuring Rules + Sanctions Flags + Geo-Risk Rules + Velocity Rules + Threshold Rules)
    ↓
STATISTICAL ANOMALY DETECTION LAYER
(Z-Score + IQR + MAD + Percentile-Based Detection)
    ↓
UNSUPERVISED & HYBRID ML ANOMALY DETECTION LAYER
(Isolation Forest + LOF + One-Class SVM + Ensemble Anomaly Scoring)
    ↓
SUPERVISED FRAUD CLASSIFICATION LAYER
(Logistic Regression + Random Forest + XGBoost)
    ↓
RISK FUSION ENGINE
(Combines Statistical + Rule-Based + ML + Behavioral Risk Scores)
    ↓
GRAPH ML FEATURE LAYER
(Customer Relationships + Shared Devices + Transaction Networks + Suspicious Connection Mapping)
    ↓
GNN FRAUD RING DETECTION LAYER
(GraphSAGE for Fraud Ring & Network Fraud Detection)
    ↓
ALERT SEVERITY ENGINE
(Low / Medium / High / Critical Risk Prioritization)
    ↓
CASE MANAGEMENT & HUMAN FEEDBACK LOOP
(Analyst Validation + False Positive Review + Investigator Notes + Audit Logging)
    ↓
→ Alerts passed to FinIntel Sentinel AI for investigation
```
---
```
│   ├── rules/               # rule-based AML engine
│   ├── statistical/         # Z-score, IQR, MAD detectors
│   ├── unsupervised/        # Isolation Forest, LOF, One-Class SVM
│   ├── supervised/          # Logistic Regression, Random Forest, XGBoost
│   ├── fusion/              # risk signal fusion engine
│   ├── gnn/                 # GraphSAGE fraud ring detection
│   ├── alerts/              # severity engine and case management
│   └── api/                 # FastAPI scoring endpoint
├── configs/
└── main.py
```
---

## Status

| Component | Status |
|---|---|
| Data Ingestion & Validation | complete |
| Feature Engineering | complete |
| Rule-Based Engine | complete |
| Statistical Detection | complete |
| Unsupervised ML | complete |
| Supervised Classification | complete |
| GNN Fraud Ring Detection | In Progress |
| Risk Fusion Engine | In Progress |
| Alert Severity Engine | In Progress |
| FastAPI Scoring API | In Progress |

---

*Part of [Financial Risk Intelligence Platform](https://github.com/Hanah7511/finrisk-intelligence-platform)*
