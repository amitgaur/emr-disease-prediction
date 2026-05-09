# EMR Disease Prediction — Ideas & Research Notes

_Auto-generated from conversation with Amit — 2026-05-08_

## Problem Statement

Build a longitudinal disease prediction system using Electronic Medical Records (EMR) data.

**Core goal:** Given a patient's historical EMR data (labs, vitals, medications, diagnoses, clinical notes), predict:
- Disease onset (e.g., Type 2 Diabetes in next 6 months)
- 30-day readmission
- ICU deterioration
- In-hospital mortality

**Longitudinal prediction** is the focus — modeling temporal sequences of patient visits/events over time.

---

## Ideas

### 1. Data Architecture

**Datasets to use:**
- **eICU-CRD** (free, ~200K ICU stays) — starting point, no credential hell
- **MIMIC-IV** (requires credentialing, gold standard) — goal after eICU
- **PubMedQA, MedQA** — for medical LLM eval
- **PhysioNet Sepsis Challenge** — focused benchmark

**Feature types:**
| Type | Examples | Processing |
|------|----------|------------|
| Structured numeric | Lab values, vitals, BMI | Normalize, handle missing |
| Diagnoses | ICD-9/10 codes | Temporal embedding |
| Medications | Drug names, dosages | Drug embedding |
| Clinical notes | Discharge summaries | BioMistral → embeddings |
| Time series | Vitals over 72hrs | GRU-D, temporal attention |

### 2. Model Architecture

**Phase 1 — Tabular Baseline (Week 1-2)**
```
XGBoost/LightGBM on aggregated patient features
Task: 30-day readmission prediction
Benchmark to beat: AUC ~0.75-0.78
```

**Phase 2 — Temporal Model (Week 3-4)**
```
FT-Transformer (Feature Tokenizer + Transformer)
Patient visit sequence → tokenized → attention over time
Target: AUC ~0.80-0.82 (SOTA for readmission)
```

**Phase 3 — LLM Enhancement (Month 2)**
```
Clinical notes → BioMistral 7B (QLoRA fine-tuned) → embeddings
Structured data → FT-Transformer → predictions
Fusion layer combines both → final prediction
```

**Phase 4 — Reasoning Layer (Month 3)**
```
Chain-of-Thought prompting on medical reasoning
Knowledge graph (ICD hierarchy, drug interactions)
RAG-style retrieval for clinical context
Multi-task: mortality + readmission + disease onset
```

### 3. SOTA Models to Explore

**Temporal Transformers:**
- FT-Transformer — attention on time series, SOTA for EHR prediction
- BEHRT — BERT on EHR sequences, disease onset prediction
- GRU-D — decay mechanisms for irregular time series
- Temporal Fusion Transformer (TFT) — multi-horizon forecasting

**LLM-based reasoning:**
- BioMistral 7B/13B — medical domain LLM, fine-tune with QLoRA
- MedBERT — encoder for clinical note embeddings
- GatorTron — NVIDIA's clinical LLM (13B+, needs beefy GPU)
- MedAgents — multi-agent reasoning system

**Hybrid approaches:**
- Structured (XGBoost) + unstructured (BioMistral) fusion
- Late fusion vs early fusion tradeoffs

### 4. Benchmarks

| Task | SOTA Model | Target AUC | Dataset |
|------|------------|------------|---------|
| 30-day readmission | FT-Transformer | ~0.80-0.82 | MIMIC-IV |
| In-hospital mortality | Multi-task LSTM, NHS-BERT | ~0.88-0.93 | MIMIC-IV |
| Sepsis prediction | InTime, TIMELY | ~0.92-0.94 | PhysioNet |
| Disease onset (T2DM) | BEHRT, MedBERT | ~0.85+ | MIMIC |
| Heart failure | GBDT + temporal | ~0.89-0.92 | MIMIC |

### 5. Tooling Stack

**CUDA/GPU:**
- GB10 available (8-16GB VRAM, Blackwell arch)
- CUDA 13.0, 119GB RAM, 3.6TB NVMe

**ML Stack:**
- PyTorch (install needed)
- Transformers + Accelerate + BitsAndBytes
- XGBoost, LightGBM
- scikit-learn
- Unsloth (QLoRA fine-tuning)
- SHAP (interpretability)

**For reasoning:**
- LangChain (RAG)
- Neo4j (knowledge graph)
- vLLM (fast inference)

### 6. Key Challenges

1. **Missing data** — patients have different lab panels done
2. **Class imbalance** — disease is rare, use focal loss or scale_pos_weight
3. **Time-based split** — never random split; train older, test newer
4. **HIPAA** — de-identify everything
5. **Interpretability** — clinicians need to trust predictions, use SHAP or attention

### 7. Learning Path Alignment

This project ties into Amit's CUDA + LLM mastery path:
- FT-Transformer training → profile with `ncu` (GPU learning)
- BioMistral QLoRA fine-tuning → CUDA kernel understanding
- Multi-GPU scaling → when moving to larger models
- KV cache optimization → inference optimization

### 8. Success Metrics

| Phase | Milestone | Metrics |
|-------|-----------|---------|
| Week 2 | XGBoost baseline | AUC > 0.75 |
| Week 4 | FT-Transformer | AUC > 0.80 |
| Week 8 | BioMistral fusion | AUC > 0.82 |
| Week 12 | Full reasoning pipeline | AUC > 0.85, CoT explainability |

---

## Next Steps

1. [ ] Install PyTorch + transformers + xgboost stack
2. [ ] Register for eICU-CRD access (https://eicu-crd.mit.edu/)
3. [ ] Scaffold data loader for eICU
4. [ ] Build XGBoost baseline — 30-day readmission
5. [ ] Profile with ncu → GPU optimization learning
6. [ ] Implement FT-Transformer
7. [ ] Fine-tune BioMistral on clinical notes
8. [ ] Build fusion layer
9. [ ] Add reasoning layer (CoT, knowledge graph)
10. [ ] Deploy via vLLM + FastAPI

---

## Related Documents

- `cuda-llm-mastery-path.md` — full learning curriculum
- `learning-interests.md` — Amit's learning goals
- `library/aleksagordic-matmul.md` — GPU architecture reference