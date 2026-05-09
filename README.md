# EMR Disease Prediction

Longitudinal disease prediction using Electronic Medical Records (EMR), temporal transformers, and medical LLM fine-tuning.

## Goal

Build a system that predicts disease onset, readmission, and ICU deterioration from patient EMR data using:

1. **Temporal Transformers** (FT-Transformer) for structured EHR sequences
2. **BioMistral 7B** (QLoRA fine-tuned) for clinical note understanding
3. **Reasoning Layer** with Chain-of-Thought prompting and knowledge graphs

## Architecture

```
Patient EMR Data
├── Structured (labs, vitals, medications, ICD codes)
│   └── FT-Transformer → temporal features
├── Unstructured (clinical notes)
│   └── BioMistral QLoRA → note embeddings
└── Time series (vitals over time)
    └── GRU-D / Temporal Attention
    
        ↓
    [Fusion Layer]
        ↓
    Prediction: disease onset, readmission, mortality
        ↓
    Reasoning: CoT explanation + KG retrieval
```

## Quick Start

```bash
# Install dependencies
pip install torch transformers xgboost lightgbm scikit-learn
pip install unsloth bitsandbytes peft accelerate shap

# Download eICU data (https://eicu-crd.mit.edu/)
# Place in data/eicu/

# Run baseline
python src/models/xgb_baseline.py

# Run temporal model
python src/models/ft_transformer.py

# Fine-tune BioMistral
python src/models/biomistral_finetune.py
```

## Project Structure

```
emr-disease-prediction/
├── docs/
│   └── ideas/           # Research notes, ideas, architecture docs
├── notebooks/           # Jupyter notebooks for exploration
├── src/
│   ├── data/           # Data loaders, preprocessing
│   ├── models/         # Model implementations
│   ├── eval/           # Benchmarking, metrics
│   └── pipeline/       # End-to-end pipeline
├── configs/             # Model configs, hyperparameters
├── data/                # EMR datasets (eICU, MIMIC)
└── results/             # Model outputs, eval results
```

## Benchmarks to Beat

| Task | Model | Target AUC |
|------|-------|------------|
| 30-day readmission | FT-Transformer | 0.80-0.82 |
| In-hospital mortality | Multi-task LSTM | 0.88-0.93 |
| Sepsis prediction | InTime, TIMELY | 0.92-0.94 |

## Hardware

- **GPU:** NVIDIA GB10 (Blackwell, 8-16GB VRAM)
- **RAM:** 119 GB
- **Storage:** 3.6TB NVMe

Optimized for inference and fine-tuning (7B-13B parameter models at 4-bit).

## Status

🚧 In progress — scaffolding the data pipeline and baselines.

## References

- FT-Transformer: Attention-based Neural Network for Time Series Forecasting
- BEHRT: BERT for Electronic Health Records
- BioMistral: Medical Domain LLM
- eICU-CRD: Multi-center ICU Database