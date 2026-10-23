# PreLN-RGCA: Pre-LayerNorm Residual Gated Cross-Stream Architecture (v1.0.0)

Official self-contained reproducibility package for:
> **Reconciling Parallel Linguistic Architectures with Deep Representation Learning: Pre-LayerNorm Gated Bilinear Cross-Stream Interaction Across Psycholinguistic and Minimal-Pair Benchmarks**  
> **Author:** Siva Phanindra Daggubati (Department of Computer Science, SGK Government Degree College, Vinukonda, Palnadu Dt., Andhra Pradesh, India; )  
> **Release URL:** [https://github.com/daggubatisivaphanindra/preln-rgca/releases/tag/v1.0.0](https://github.com/daggubatisivaphanindra/preln-rgca/releases/tag/v1.0.0)

## Quickstart & One-Command Reproducibility (Relative Paths Only)

From the root of the extracted repository ():

```bash
# 1. Run unit & integration tests (PAD invariance, analytic gradients, frame isolation, ablation isolation)
python3 tests/test_preln_rgca.py

# 2. Run the full 10-seed experimental suite (generates splits, checkpoints, CSVs)
python3 scripts/run_all_experiments.py

# 3. Independently validate all saved CSVs, frame splits, surface stimuli, and .npz checkpoints
python3 scripts/validate_results.py
```

## Key Methodological & Audit Guarantees

1. **Modular Architecture (`src/`) & Relative Paths**: All model components (`src/model.py`), dataset builders and training-only SVD embedding estimators (`src/data.py`), symbolic rule baselines (`src/baselines.py`), and statistical estimators (`src/metrics.py`) use relative paths resolved from the repository root.
2. **Strict Zero Lexical-Frame Leakage (`splits/*.csv`)**: Entire lexical frames (`grouping_id`) are partitioned into disjoint `train`, `val`, and `test` sets prior to item selection (`0` frames appearing in multiple splits across ON99, KO05, BLiMP, CoLA, and the External Naturalistic Benchmark).
3. **Training-Only Distributional SVD Embeddings (`fit_training_only_svd_embeddings`)**: Word representations are estimated exclusively from the training partition (`tr_idx`) of each run without prescribed semantic category lists (`ANIMATE_NOUNS` / `INANIMATE_NOUNS` are never used to construct embeddings) and with zero access to validation or test splits. We also report a **Strict Out-of-Vocabulary (OOV) Disjoint-Lexeme Control** where unseen test nouns receive zero training exposure.
4. **Inflected Surface Stimuli & Deterministic Tag Agreement**: Every sentence uses valid inflected English surface forms (`eating` vs `eaten`, `spilling` vs `spilled`, `was` vs `were` vs `will` vs `can`). Morphosyntactic tags are extracted deterministically from the surface words (`0` surface-to-tag mismatches, `0` conflicting labels on identical displayed sequences).
5. **Alignment-Matched Symbolic Baseline**: Directly uses the exact same clause-local head-subject syntactic alignment matrix `A_struct` as PreLN-RGCA to evaluate whether the proposed neural interaction provides benefits beyond the hand-engineered structural alignment alone.
6. **Strictly Unconfounded Single-Mechanism Ablations (`src/model.py`)**: Every ablation changes strictly one architectural mechanism while holding initialization scale (`init_mult = 1.0`), alignment matrices (`P_cross = A_struct` except in `ablation_fixed_align` which tests `P_fix`), and optimizer settings 100% constant.
7. **Actual Saved Checkpoints (`checkpoints/*.npz`)**: All 50 trained full-model checkpoints (5 benchmarks x 10 seeds) are saved as compressed NumPy `.npz` archives whose SHA-256 hashes match `results/proposed/seed_runs.csv`.
