# EMR Disease Prediction — Architecture & Moat Strategy

_Last updated: 2026-05-08_

## Core Insight: The Moat Isn't the Model

Fine-tuning BioMistral is table stakes — anyone with a GPU and MIMIC can do it. Real defensibility comes from:

1. **Proprietary data curation** (hardest to replicate)
2. **Architecture innovations** (medical-specific)
3. **Training recipe** (clinical CoT, uncertainty, curriculum)
4. **Inference moats** (uncertainty-aware, counterfactual)
5. **Integration** (EHR workflows, not just API)

---

## Moat Builders

### Moat #1: Temporal Knowledge Graph Injection

Pre-load ICD hierarchy, drug-drug interactions into attention mechanism.

```python
class DiseaseGraphAttention(nn.Module):
    """
    Inject ICD hierarchy into embedding space.
    'E11.9' (Type 2 DM no complications) → related to E11 (Type 2 DM)
    Model knows disease relationships structurally.
    Other models learn this from data; you bake it in.
    """
    def __init__(self, icd_graph):
        self.graph = icd_graph  # Pre-loaded ICD hierarchy
        # Attention weights modulated by graph distance
```

### Moat #2: Medical-Specific Attention Decay

Different lab values decay at different rates:
- Heart rate: fast decay (current value matters most)
- HbA1c: slow decay (3-month average matters)
- Creatinine: medium (weekly trends matter)

```python
def medical_attention(query, key, value, lab_decay_rates):
    # Lab values decay at different rates
    attention = standard_attention(query, key, value)
    return attention * decay_mask  # Decay based on lab type
```

### Moat #3: Uncertainty-Aware Predictions

Train with conformal prediction. Model outputs:

```python
{
    "risk": 0.23,
    "uncertainty": "high",  # Model knows it's uncertain
    "similar_patients_confidence": 0.4,
    "recommendation": "consult_physician"
}
```

Clinicians trust this more than a flat probability.

### Moat #4: Counterfactual Reasoning

```python
def counterfactual_explanation(patient, model, target_risk_delta=-0.1):
    """
    'If patient's HbA1c was 6.5 instead of 7.8, 
     risk would decrease by 18%'
    """
    perturbed = patient.copy()
    perturbed.labs["HbA1c"] = 6.5
    return model.predict_risk_delta(patient, perturbed)
```

### Moat #5: Multi-Task Curriculum Learning

Train on tasks in order:
1. Simple: 30-day readmission
2. Medium: in-hospital mortality  
3. Hard: disease onset (6 months)

Each task builds temporal representations for the next.

```python
tasks = ["readmission", "mortality", "disease_onset"]
for i, task in enumerate(tasks):
    if i > 0:
        freeze_lower_layers()  # Retain prior knowledge
    train(task, curriculum_order=i)
```

### Moat #6: Continuous Learning

Static benchmark models degrade. This adapts:

```python
class ContinuousLearner:
    def __init__(self, model, update_frequency_days=7):
        self.model = model
        self.update_freq = update_frequency_days
        
    def weekly_update(self, new_patients, outcomes):
        """
        Retrain on new outcomes without full retraining.
        Uses replay buffer to avoid catastrophic forgetting.
        """
        self.model.fine_tune_on_recent(new_patients, outcomes)
        self.model.evict_old_representation()
```

### Moat #7: Multi-Modal Fusion

Public models are unimodal. Most powerful approach:

```python
class MultiModalFusion(nn.Module):
    """
    Combine imaging + labs + notes in one model.
    Most open-source work is text-only or tabular-only.
    """
    def __init__(self):
        self.image_encoder = load("chext-xray-vit")
        self.lab_encoder = FT_Transformer(layers=4)
        self.note_encoder = load("BioMistral-7B")
        self.fusion = CrossAttentionFusion()
```

---

## Architecture (Full Pipeline)

```
Patient EMR Data
│
├── Structured (labs, vitals, medications, ICD codes)
│   ├── Temporal Knowledge Graph (injected into attention)
│   ├── Lab-specific decay rates
│   └── FT-Transformer → temporal features
│
├── Unstructured (clinical notes)
│   ├── BioMistral 7B QLoRA → note embeddings
│   └── Multi-task (readmission → mortality → onset)
│
├── Time series (vitals over time)
│   └── GRU-D with medical decay
│
└── Knowledge Graph
    ├── ICD hierarchy
    ├── Drug-drug interactions
    └── Clinical guidelines (AHA, ADA rules)
         │
         ▼
    [Fusion Layer]
         │
    Prediction: disease onset, readmission, mortality
         │
    Reasoning Layer
    ├── Chain-of-Thought (CoT) explanation
    ├── Uncertainty quantification
    ├── Counterfactual reasoning
    └── Similar patient retrieval
```

---

## Success Metrics

| Moat | Metric | Target |
|------|--------|--------|
| Uncertainty | % predictions with valid uncertainty | >80% |
| Counterfactual | Clinician acceptance of explanations | >60% |
| Continuous learning | AUC drift over 6 months | <0.02 |
| Multi-modal | Improvement over text-only | AUC +0.05 |
| Knowledge graph | Improvement over raw attention | AUC +0.03 |

---

## References

- Temporal Fusion Transformer (TFT) for EHR
- BEHRT: BERT for Electronic Health Records  
- MedAgents: Multi-Agent Medical Reasoning
- Conformal Prediction for uncertainty quantification
- ICD-9/ICD-10 hierarchy graph structures