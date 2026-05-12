# CUDA + Medical LLM Learning Curriculum

_7 sessions, 2-4 hours each. Hands-on, tied to emr-disease-prediction._

---

## Session 1: CUDA on Blackwell — First Kernel & Profiling

**Objective:** Write, compile, and profile your first CUDA kernel on GB10 to understand the Blackwell execution model from bare metal up.

### Topics

- CUDA execution model: grids, blocks, warps, SM occupancy
- GB10 Blackwell architecture: SM count, shared memory size, L2 cache, sm_121
- Memory hierarchy: global -> L2 -> shared -> registers
- `nvcc` compilation pipeline: `.cu` -> PTX -> SASS
- Nsight Compute (`ncu`) profiling: memory throughput, warp occupancy, stall reasons
- PyTorch 2.10 on GB10: CPU-only limitation (sm_121 not in pre-built wheels), compiling from source vs. using custom kernels directly

### Key Repos to Study

| Repo | Focus Files/Functions |
|------|----------------------|
| `rust-gpu/rust-cuda` | `guide/src/` — CUDA programming model explainer |
| NVIDIA CUDA Samples | `0_Introduction/vectorAdd/vectorAdd.cu` — canonical first kernel |
| NVIDIA `ncu` docs | Roofline analysis model, memory chart interpretation |

### Hands-On Exercise (GB10)

1. **Vector add in C/CUDA:**
   ```bash
   # Write vectorAdd.cu with 1M floats
   nvcc -arch=sm_121 -o vectorAdd vectorAdd.cu
   ncu --set full ./vectorAdd
   ```
2. **Reduction kernel:** Sum 10M floats. First naive (one thread per element), then shared-memory reduction with `__syncthreads()`. Compare `ncu` memory throughput between versions.
3. **Inspect PTX:** `nvcc -ptx -arch=sm_121 vectorAdd.cu` — read the generated PTX, identify load/store instructions, understand register allocation.
4. **Benchmark vs NumPy:** Time the reduction against `np.sum()` on CPU. Measure PCIe transfer overhead with `ncu` to understand when GPU wins.

### Connection to EMR Project

Your `FTTransformerEHR` (`src/models/ft_transformer.py:111`) runs attention matmuls on GPU. Understanding warp scheduling and memory coalescing tells you why batch size, sequence length, and `d_model` choices affect throughput. The `FeatureTokenizer` loop at line 52 (`for t in range(seq_len)`) is a sequential bottleneck — by end of session you'll see why and what a fused kernel could do.

### Final Outcome

You can write a CUDA kernel in C, compile for Blackwell sm_121, profile with `ncu`, and read a roofline chart. You understand why your FT-Transformer's GPU utilization varies with batch size.

**Time: ~2.5 hours**

---

## Session 2: Rust-CUDA Ecosystem — cuda-oxide, cutile-rs, rust-cuda

**Objective:** Evaluate whether Rust replaces C for GPU kernel development on Blackwell by building the same kernels from Session 1 in Rust and comparing ergonomics, safety, and performance.

### Topics

- Rust-to-PTX compilation: how `cuda-oxide` skips `nvcc` entirely
- Type-safe GPU memory management: `DeviceBuffer<T>` vs raw `cudaMalloc`
- `cutile-rs` tile abstraction: declarative kernel shapes vs manual indexing
- `oxi-cuda` (oxicuda): full CUDA toolkit replacement, autotuner for Blackwell
- Trade-offs: Rust compile times, ecosystem maturity, debugging story
- When to use Rust vs C: new kernels vs modifying existing CUDA codebases

### Key Repos to Study

| Repo | Focus Files/Functions |
|------|----------------------|
| `NVlabs/cuda-oxide` | `examples/` — vector_add, reduction in safe Rust; `src/compiler/` — PTX codegen pipeline |
| `NVlabs/cutile-rs` | `src/tile.rs` — tile abstraction; `examples/gemm.rs` — tiled matmul |
| `oxicuda/oxicuda` | `src/runtime/` — CUDA runtime replacement; `src/autotuner/` — Blackwell autotuning |
| `rust-gpu/rust-cuda` | `crates/cuda_std/src/` — GPU intrinsics (`thread::index`, `shared_array!`); `examples/` — PTX generation |

### Hands-On Exercise (GB10)

1. **Port vector_add to cuda-oxide:**
   ```rust
   // Same 1M float add from Session 1
   // Compare: DeviceBuffer<f32> vs cudaMalloc + pointer casting
   // Compile to PTX, inspect output
   ```
2. **Port reduction to cutile-rs:** Use the tile DSL to express shared-memory reduction. Compare generated PTX with hand-written C version from Session 1.
3. **Benchmark head-to-head:** Same kernel, same data, C vs Rust. Profile both with `ncu`. Document: latency, memory throughput, code lines, compile time.
4. **Try oxi-cuda autotuner:** Let it search block sizes for your reduction kernel on GB10. Compare its choice vs your manual pick.

### Connection to EMR Project

If Rust kernels match C performance with better safety, custom attention kernels for `TemporalAttentionLayer` (`src/models/layers.py:31`) can be written in Rust. The `GatedResidualConnection` (`layers.py:54`) fuses linear + GELU + gate + norm — a custom fused kernel eliminates intermediate memory reads. Rust's type system prevents the shape mismatches that plague hand-written CUDA.

### Final Outcome

You have a decision framework: Rust vs C for your project's custom kernels. You've written the same kernel in both, profiled both on Blackwell, and know the trade-offs cold.

**Time: ~3 hours**

---

## Session 3: Blackwell Matmul Mastery — TMA, WGMMA, Warp Tiling

**Objective:** Implement a high-performance matrix multiply on Blackwell using Tensor Memory Accelerator (TMA) and Warp Group Matrix Multiply-Accumulate (WGMMA) — the two instructions that define Blackwell kernel performance.

### Topics

- Why matmul is the bottleneck: attention is `O(n^2 * d)` matmul
- Naive matmul -> tiled matmul -> warp-tiled matmul progression
- TMA (Tensor Memory Accelerator): async bulk copies, no manual address math
- WGMMA: warp group (4 warps = 128 threads) cooperative matmul on Tensor Cores
- Warp tiling strategy: CTA tile -> warp group tile -> thread tile
- Register pressure management on Blackwell (255 registers/thread)
- Async pipeline: TMA prefetch overlaps with WGMMA compute

### Key Repos to Study

| Repo | Focus Files/Functions |
|------|----------------------|
| Aleksa Gordic matmul blog (`library/aleksagordic-matmul.md`) | Full walkthrough: naive -> shared mem -> register tiling -> Tensor Core; Hopper TMA + WGMMA explanation |
| `NVlabs/cutile-rs` | `examples/gemm.rs` — tiled GEMM with Rust tile DSL |
| Colfax Research GEMM kernels | TMA descriptor setup, WGMMA instruction encoding, software pipelining |
| CUTLASS 3.x | `include/cute/` — CuTe layout algebra; `examples/` — Hopper/Blackwell GEMM |

### Hands-On Exercise (GB10)

1. **Naive matmul (C):** 1024x1024 FP16 multiply. Profile with `ncu` — note DRAM bandwidth utilization (expect ~5-10% of peak).
2. **Shared memory tiled:** 64x64 tiles, `__shared__` staging. Re-profile — should hit ~30-50% bandwidth.
3. **WGMMA version:** Use `mma.sync` / WGMMA PTX intrinsics for 16x16 tiles on Tensor Cores. Measure TFLOPS.
4. **TMA prefetch pipeline:** Add TMA-based async loads with double-buffering. Overlap load of tile N+1 with compute of tile N.
5. **Compare against cuBLAS:** `cublasGemmEx` on same sizes. Your best kernel vs NVIDIA's — gap tells you where to focus.

### Connection to EMR Project

Every forward pass of `FTTransformerEHR` runs matmuls in `nn.MultiheadAttention` (`layers.py:48`). With 4 attention heads, 4 layers, batch=64, seq_len=64, d_model=128: that's 16 matmuls per forward pass. Understanding TMA+WGMMA tells you exactly what cuBLAS is doing under the hood and whether a fused attention kernel (FlashAttention-style) makes sense for your sequence lengths.

### Final Outcome

You can write a Blackwell matmul kernel using TMA + WGMMA that reaches >50% of cuBLAS throughput. You understand the warp tiling hierarchy and can read any modern GEMM kernel.

**Time: ~4 hours**

---

## Session 4: FT-Transformer Deep Dive — Temporal EHR Attention

**Objective:** Extend your existing FT-Transformer with medical-domain attention innovations (decay-aware attention, temporal position encoding) and validate with time-based cross-validation on synthetic patient data.

### Topics

- FT-Transformer architecture review: FeatureTokenizer -> temporal self-attention -> classification head
- Medical attention decay: different lab values have different half-lives (HbA1c ~90 days, heart rate ~minutes)
- Temporal position encoding: sinusoidal (current) vs learnable continuous-time vs log-scaled elapsed time
- Multi-task curriculum learning: readmission -> mortality -> disease onset (your `MultiTaskModel` at `src/models/multi_task.py`)
- Time-based cross-validation: expanding vs sliding window (your `TimeBasedSplit` at `src/eval/temporal_split.py`)
- Calibration: ECE, reliability diagrams (your `calibration.py`)

### Key Repos to Study

| Repo | Focus Files/Functions |
|------|----------------------|
| Your codebase | `src/models/ft_transformer.py` — `FTTransformerEHR`, `FeatureTokenizer`; `src/models/layers.py` — `TemporalAttentionLayer`, `GatedResidualConnection`; `src/models/multi_task.py` — `MultiTaskModel`, `MultiTaskLoss._get_curriculum_weight` |
| Your codebase | `src/eval/temporal_split.py` — `expanding_window_split`, `TimeBasedSplit`; `src/eval/calibration.py` — `expected_calibration_error`, `plot_reliability_diagram` |
| `pytorch-forecasting` | `temporal_fusion_transformer.py` — variable selection network, interpretable attention |
| FT-Transformer paper | Gorishniy et al. — feature tokenization for tabular data |

### Hands-On Exercise (GB10)

1. **Generate synthetic eICU-like data:** 10K patients, 20 lab features, variable-length sequences (5-50 visits). Use your `EICULoader` schema (`src/data/eicu_loader.py`) as the template.
2. **Add medical attention decay to `TemporalAttentionLayer`:**
   ```python
   # In layers.py: modulate attention weights by lab-specific decay
   # HbA1c (slow decay, tau=90 days), creatinine (medium, tau=7 days),
   # heart_rate (fast, tau=0.5 days)
   decay_mask = torch.exp(-time_delta / tau_per_feature)
   attn_weights = attn_weights * decay_mask
   ```
3. **Train multi-task model:** Use `MultiTaskModel` with curriculum (readmission at epoch 0, mortality at epoch 5, disease_onset at epoch 10). Log per-task loss curves.
4. **Evaluate with temporal CV:** Use `TimeBasedSplit(mode="expanding", n_splits=5)`. Plot calibration curves per fold with `plot_reliability_diagram`. Compute ECE.
5. **Profile the training loop:** `torch.profiler` or `ncu` on a single forward pass. Identify the bottleneck (likely the sequential `for t in range(seq_len)` loop in `FeatureTokenizer.forward`).

### Connection to EMR Project

This IS the core of your EMR project. The medical attention decay is Moat #2 from your `MOAT_STRATEGY.md`. The curriculum learning is Moat #5. After this session, your FT-Transformer is no longer generic — it encodes medical domain knowledge about how lab values age.

### Final Outcome

You have a medical-domain FT-Transformer with decay-aware attention and curriculum multi-task learning, validated with temporal CV and calibration analysis. You know exactly where the training bottleneck is.

**Time: ~3.5 hours**

---

## Session 5: Medical LLM Fine-Tuning — BioMistral 7B + QLoRA

**Objective:** Fine-tune BioMistral 7B with QLoRA on clinical note snippets using Unsloth on GB10, producing note embeddings that plug into your fusion pipeline.

### Topics

- BioMistral 7B architecture: Mistral base + biomedical pre-training
- QLoRA mechanics: 4-bit NF4 quantization + low-rank adapters (rank 16-64)
- Unsloth: 2x faster fine-tuning, patched attention, RoPE scaling
- Training recipe: clinical note -> structured extraction (ICD codes, risk factors, temporal markers)
- Embedding extraction: last hidden state vs mean pooling vs task-specific [CLS]
- GB10 constraints: ~8GB VRAM for 7B at 4-bit + LoRA gradients; batch size 1-2 with gradient accumulation

### Key Repos to Study

| Repo | Focus Files/Functions |
|------|----------------------|
| `unsloth/unsloth` | `models/mistral.py` — patched Mistral attention; `tokenizer_utils.py` — chat template setup |
| `BioMistral/BioMistral-7B` (HuggingFace) | Model card — pre-training data (PubMed, clinical trials); tokenizer — medical vocabulary coverage |
| CARE-AD paper (npj Digital Medicine) | Multi-agent architecture: Screening Agent, Diagnostic Agent, Treatment Agent — each a fine-tuned LLM with different clinical note views |
| Your codebase | `README.md` — `src/models/biomistral_finetune.py` (referenced, to be built) |

### Hands-On Exercise (GB10)

1. **Setup Unsloth + BioMistral:**
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       "BioMistral/BioMistral-7B",
       max_seq_length=2048,
       load_in_4bit=True,
   )
   model = FastLanguageModel.get_peft_model(model, r=16, target_modules=[
       "q_proj", "k_proj", "v_proj", "o_proj",
       "gate_proj", "up_proj", "down_proj",
   ])
   ```
2. **Create training data:** Format 500 synthetic clinical notes as instruction-response pairs:
   ```
   [INST] Extract diagnoses, risk factors, and temporal markers from this note:
   "72yo M with h/o HTN, DM2 presents with chest pain x 2 days..."
   [/INST]
   Diagnoses: I10 (HTN), E11.9 (DM2), R07.9 (chest pain)
   Risk factors: age>65, male, hypertension, diabetes
   Temporal: symptom onset 2 days prior to admission
   ```
3. **Fine-tune with Unsloth:** 3 epochs, lr=2e-4, batch_size=2, gradient_accumulation=8. Monitor loss curve and VRAM usage.
4. **Extract embeddings:** Run inference on 100 notes, extract last hidden state (mean-pooled). Save as `(100, 4096)` tensor.
5. **Sanity check:** Cluster embeddings with UMAP. Do similar diagnoses cluster together? Compare raw BioMistral vs fine-tuned embeddings.

### Connection to EMR Project

These note embeddings become the "Unstructured" branch of your architecture (`README.md` pipeline diagram). They'll fuse with FT-Transformer features in Session 7. The CARE-AD multi-agent pattern from the paper maps directly to your multi-task setup: different agents for different prediction tasks, each seeing clinical notes through a different lens.

### Final Outcome

You have a QLoRA-fine-tuned BioMistral 7B that extracts structured information from clinical notes and produces medical-domain embeddings. You know the GB10 VRAM budget for inference and fine-tuning at 4-bit.

**Time: ~3.5 hours**

---

## Session 6: Multi-Agent Clinical Reasoning — CARE-AD + Knowledge Graphs

**Objective:** Implement the reasoning layer from your moat strategy — knowledge graph injection, multi-agent clinical reasoning (CARE-AD pattern), and uncertainty quantification via conformal prediction.

### Topics

- CARE-AD architecture: Screening -> Diagnostic -> Treatment agents, each processing longitudinal notes
- Temporal knowledge graph: ICD-9/10 hierarchy as attention bias (Moat #1 from `MOAT_STRATEGY.md`)
- Drug-drug interaction graph: RxNorm relationships as edge features
- Conformal prediction: distribution-free uncertainty sets, coverage guarantees
- Counterfactual reasoning: perturbation-based explanations (Moat #4)
- Multi-agent orchestration: LangGraph or lightweight custom pipeline

### Key Repos to Study

| Repo | Focus Files/Functions |
|------|----------------------|
| CARE-AD paper code | Agent definitions, note chunking by time window, diagnostic reasoning chain |
| Your `MOAT_STRATEGY.md` | `DiseaseGraphAttention` (Moat #1), `medical_attention` decay (Moat #2), conformal prediction (Moat #3), `counterfactual_explanation` (Moat #4) |
| `spark-engine-ai/LAuRA` | `src/agents/` — multi-agent orchestration pattern; `src/planner/` — task decomposition; transferable to clinical agent pipeline |
| `mapie` (Python) | `MapieClassifier` — conformal prediction wrapper for sklearn/PyTorch classifiers |

### Hands-On Exercise (GB10)

1. **Build ICD hierarchy graph:**
   ```python
   # Parse ICD-10 tree into networkx graph
   # E11 -> E11.0, E11.1, ..., E11.9 (Type 2 DM subtypes)
   # Compute shortest-path distances for attention bias
   icd_graph = build_icd_hierarchy("icd10cm_2026.xml")
   distance_matrix = shortest_path_distances(icd_graph)
   ```
2. **Implement `DiseaseGraphAttention`:** Modulate attention scores by ICD graph distance. Diseases closer in the hierarchy attend to each other more strongly.
3. **Conformal prediction wrapper:**
   ```python
   # Wrap your MultiTaskModel predictions with conformal sets
   from mapie.classification import MapieClassifier
   # Calibrate on held-out temporal fold
   # Output: prediction sets with guaranteed coverage
   ```
4. **Build CARE-AD-style pipeline:** 3 agents (Screening, Diagnostic, Risk) each calling your fine-tuned BioMistral. Chain outputs with structured prompts.
5. **Counterfactual reasoning:** For a synthetic patient, perturb one feature at a time (HbA1c: 7.8 -> 6.5). Show risk delta and generate natural language explanation.

### Connection to EMR Project

This session implements Moats #1, #3, and #4 from your `MOAT_STRATEGY.md`. The ICD graph attention goes into `TemporalAttentionLayer`. Conformal prediction wraps your `MultiTaskModel.predict()`. The CARE-AD agent pattern becomes your reasoning layer. The LAuRA multi-agent orchestration pattern transfers directly — same concept (multiple specialized agents), different domain.

### Final Outcome

You have a knowledge-graph-aware attention mechanism, conformal prediction with coverage guarantees, counterfactual explanations, and a multi-agent clinical reasoning pipeline. These are the moats that differentiate your system from "fine-tuned model + API."

**Time: ~3.5 hours**

---

## Session 7: Full Pipeline — Multi-Modal Fusion, Demo & Profiling

**Objective:** Wire everything together into an end-to-end pipeline (structured EHR + clinical notes + knowledge graph -> prediction + explanation), profile the full system on GB10, and build a demo-ready interface.

### Topics

- Cross-attention fusion: FT-Transformer features (d=128) + BioMistral embeddings (d=4096)
- Dimension alignment: projection layers, feature-level vs sequence-level fusion
- End-to-end training: frozen BioMistral embeddings + trainable FT-Transformer + fusion layer
- System profiling: `nsys` for full-pipeline latency, identify CPU/GPU bottlenecks
- Custom CUDA kernels in the pipeline: where Rust/C kernels from Sessions 1-3 slot in
- Demo UI with LAuRA: React frontend showing patient timeline, predictions, explanations

### Key Repos to Study

| Repo | Focus Files/Functions |
|------|----------------------|
| Your codebase | All of `src/models/` — `ft_transformer.py`, `multi_task.py`, `layers.py`; `src/eval/` — `calibration.py`, `temporal_split.py` |
| Your `MOAT_STRATEGY.md` | `MultiModalFusion` class (Moat #7), `ContinuousLearner` (Moat #6) |
| `spark-engine-ai/LAuRA` | `src/builder/` — React component generation; `src/agents/ui_agent.py` — UI planning from natural language spec |
| FlashAttention | `flash_attn/flash_attn_triton.py` — fused attention pattern applicable to your `TemporalAttentionLayer` |

### Hands-On Exercise (GB10)

1. **Build `MultiModalFusion` module:**
   ```python
   class MultiModalFusion(nn.Module):
       def __init__(self, ehr_dim=128, note_dim=4096, fused_dim=256):
           self.ehr_proj = nn.Linear(ehr_dim, fused_dim)
           self.note_proj = nn.Linear(note_dim, fused_dim)
           self.cross_attn = nn.MultiheadAttention(fused_dim, num_heads=8)
           self.classifier = nn.Linear(fused_dim, 3)  # readmission, mortality, onset
   ```
2. **End-to-end forward pass:** Synthetic patient with 20 visits + 3 clinical notes. FT-Transformer processes visits, BioMistral processes notes (pre-computed embeddings), fusion layer combines. Measure latency on GB10.
3. **Profile with `nsys`:**
   ```bash
   nsys profile --stats=true python pipeline_demo.py
   # Identify: data loading vs FT-Transformer vs BioMistral vs fusion
   ```
4. **Insert custom kernel:** Replace the `FeatureTokenizer` sequential loop (`ft_transformer.py:52`) with a batched kernel (Rust or C) from Sessions 1-2. Measure speedup.
5. **Build demo with LAuRA:**
   - Patient timeline visualization (visits, labs, medications over time)
   - Risk predictions with uncertainty bars (conformal intervals)
   - Counterfactual panel: "What if HbA1c was 6.5?"
   - Agent reasoning trace: Screening -> Diagnostic -> Risk agent chain

### Connection to EMR Project

This is the culmination. Your `README.md` architecture diagram becomes a running system. The structured branch (FT-Transformer) and unstructured branch (BioMistral) merge in the fusion layer. Knowledge graph attention and conformal prediction wrap the outputs. LAuRA builds the demo UI you'll show to leaders at Function Health.

### Final Outcome

You have a running multi-modal EMR prediction pipeline on GB10: structured EHR data + clinical notes -> temporal transformer + medical LLM -> fused predictions with uncertainty + counterfactual explanations + agent reasoning. You can profile every component, identify bottlenecks, and swap in custom CUDA kernels. You have a React demo ready to show.

**Time: ~4 hours**

---

## Curriculum Summary

| # | Session | Hours | Builds On | Key Deliverable |
|---|---------|-------|-----------|-----------------|
| 1 | CUDA on Blackwell | 2.5 | — | First kernel + ncu profiling |
| 2 | Rust-CUDA Ecosystem | 3 | Session 1 | Rust vs C decision with benchmarks |
| 3 | Blackwell Matmul | 4 | Sessions 1-2 | TMA + WGMMA kernel at >50% cuBLAS |
| 4 | FT-Transformer Deep Dive | 3.5 | Session 3 | Medical decay attention + temporal CV |
| 5 | BioMistral Fine-Tuning | 3.5 | — | QLoRA fine-tuned note embeddings |
| 6 | Multi-Agent Reasoning | 3.5 | Sessions 4-5 | Knowledge graph + conformal + CARE-AD agents |
| 7 | Full Pipeline & Demo | 4 | All | End-to-end system + React demo |

**Total: ~24 hours (7 sessions)**

## Prerequisites Checklist

- [ ] GB10 with CUDA 13.x+ installed (`nvcc --version`, `nvidia-smi`)
- [ ] Rust toolchain (`rustup`, nightly for GPU targets)
- [ ] `ncu` and `nsys` (Nsight Compute / Nsight Systems)
- [ ] Python 3.12 + venv with project dependencies
- [ ] Clone repos: `cuda-oxide`, `cutile-rs`, `oxicuda`, `rust-cuda`, `LAuRA`
- [ ] eICU data access (or synthetic data generator for Sessions 4-7)

## After Completing All Sessions

You will be able to:

1. **Write and profile CUDA kernels** in both C and Rust for Blackwell (sm_121)
2. **Explain TMA, WGMMA, and warp tiling** and when each matters for attention workloads
3. **Train an FT-Transformer** on patient visit sequences with medical-domain attention decay
4. **Fine-tune BioMistral 7B** with QLoRA on clinical notes using Unsloth
5. **Build a multi-agent clinical reasoning pipeline** following the CARE-AD pattern
6. **Quantify prediction uncertainty** with conformal prediction and generate counterfactual explanations
7. **Fuse structured + unstructured EMR data** in a multi-modal architecture
8. **Demo the full system** with a React UI showing predictions, uncertainty, and reasoning traces
