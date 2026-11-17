# Fault diagnosis of Ethernet physical-layer signals with CEEMDAN features, GLF-Encoding and VarCon

Code, data, trained checkpoints and logs for four experiments.

| Folder | Experiment / Stage | Training / Script | Evaluation without training |
|---|---|---|---|
| `00_preprocessing/` | Raw-to-feature preprocessing | `01_ceemdan_decompose.py` → `03_split_and_select_features.py` | — (writes to `00_preprocessing/work/`) |
| `01_proposed/` | Proposed method | `train_proposed.ipynb` | `evaluate_checkpoint.ipynb` |
| `02_comparison/` | 7 tabular baselines | `train_baselines.ipynb` | `evaluate_checkpoints.ipynb` |
| `03_ablation/` | Ablation (6 configurations) | `train_ablation.ipynb` | `evaluate_checkpoints.ipynb` |
| `04_sensitivity/` | Parameter sensitivity (4 parameters × 5 values) | — (see below) | `evaluate_sensitivity.ipynb` |

`common/core.py` holds the data checks, the preprocessing and the model classes. The three evaluation notebooks (`01`, `03`, `04`) import it for `prepare`, `make_encoder`, `DownstreamClassifier`, `PureBaselineMLP`, `ABL`, `DEVICE` and, in `01`, `full_metrics`; nothing in the package calls its training routine. Training lives in the training notebooks, which are self-contained (same models and protocol, each with its own config block), so each folder can be read on its own.

`02_comparison/train_baseline.py` holds the seven baselines and the `evaluate()` helper; the 02 training notebook and the 02 evaluation notebook both use it.

## Data (`data/`)

Window-level features of four classes (Aging, Fault, Interference, Normal). The class called *Interference* in the paper is stored as `Noise` in the CSV files.

| Folder | Features | train / val / test rows |
|---|---|---|
| `features_21d/` | 21 CEEMDAN features (3 statistics of the raw signal + 3 per IMF for 6 IMFs) | 197,001 / 21,000 / 62,002 |
| `features_11d/` | 11 statistics of the raw waveform (no CEEMDAN), used by *Baseline* and *With M3* | same rows, same split |

`data/MD5SUMS.txt` lists the file checksums. Every training/evaluation notebook verifies size, md5, sha256, row and class counts and a content digest of the feature matrix before running, and executes a negative self-test (a copy with one value changed by 1e-9 must be rejected).

Raw continuous oscilloscope waveforms and intermediate window-level CEEMDAN CSV files are omitted to keep the release compact; `00_preprocessing/` contains the reference scripts for sliding-window CEEMDAN decomposition, 77-dimensional time-frequency feature extraction, 2,500-window guard-band splitting, training-only standardization, and intra-group linear-SVM RFE. Feature standardization was fitted exclusively on the training split (`val` and `test` were transformed using training statistics), and the CSV metadata columns (`sample_id`, `window_number`, `source_folder`, `source_sample`) preserve the original acquisition identifiers.

Across the two feature sets, all `val` (21,000) and `test` (62,002) rows match one-to-one; in `train` (197,001), 196,999 rows match one-to-one while the two duplicate-acquisition records of `Aging` window `49926` (`Aging` and `PAging` at 1-based CSV lines 39678–39679) appear in swapped order between `features_11d/` and `features_21d/` (each experiment reads only one feature set).

## Results (seed 163, test set of 62,002 windows)

### Comparison

| Model | Parameters | Test accuracy | Macro-F1 |
|---|---:|---:|---:|
| TabNet | 1,076 | 98.08% | 0.9808 |
| FT-Transformer | 4,044 | 98.75% | 0.9875 |
| AutoInt | 4,658 | 98.82% | 0.9882 |
| DCN-V2 | 3,997 | 98.45% | 0.9844 |
| ExcelFormer | 16,100 | 98.40% | 0.9840 |
| Tab-ResNet | 18,948 | 97.95% | 0.9794 |
| NODE | 2,404 | 98.90% | 0.9889 |
| **Proposed** | 110,660 (516 trainable in phase 2) | **99.45%** | 0.9945 |

*Note on baseline architectures:* All seven baselines in `02_comparison/train_baseline.py` are compact PyTorch adaptations (~1k–19k parameters) built around each method's core structural motif for the 21-dimensional tabular input (e.g., single-step sigmoid masking + MLP in `TabNetModel`, feature-level multi-head attention + gated FFN in `ExcelFormerModel`, single-layer entmax oblivious trees in `NODEModel`) and trained under a unified optimization protocol rather than full heavy original packages.

### Ablation

M1 = CEEMDAN feature processing, M2 = GLF-Encoding, M3 = VarCon contrastive learning.

| Configuration | Test accuracy | Macro-F1 |
|---|---:|---:|
| Baseline (11-dim raw-waveform features + MLP) | 85.42% | 0.8526 |
| With M1 (21-dim CEEMDAN features + MLP) | 87.56% | 0.8740 |
| With M3 | 87.62% | 0.8742 |
| With M1+M2 | 99.01% | 0.9901 |
| With M1+M3 | 99.04% | 0.9904 |
| Proposed | 99.45% | 0.9945 |

*With M1+M2* pretrains the GLF-Encoding module with cross-entropy instead of VarCon. *With M3* and *With M1+M3* replace GLF-Encoding with an MLP encoder. All other settings are the proposed method's.

### Sensitivity

`04_sensitivity/points.csv` and `fig_sensitivity.png`: test accuracy of the proposed method when one of ρ (feature keep ratio), τ₁ (temperature), β (consistency weight) or α (centroid momentum) is changed and the others stay at their defaults (ρ 0.9, τ₁ 0.07, β 0.25, α 0.9).

## Reproducing

```bash
pip install -r requirements.txt
```

Run any notebook from its own folder (paths are relative). Training notebooks write to `results/` next to the notebook; the released outputs are in `released_results/` and are not overwritten.

- The four `evaluate_*` notebooks run either on a GPU or on a CPU and never train anything, so re-checking the released numbers takes minutes. They load a checkpoint with `map_location="cpu"` and then move the model to `common.core.DEVICE`, which is CUDA when a visible GPU is present and CPU otherwise (set `CUDA_VISIBLE_DEVICES=""` to force the CPU path). Every notebook prints the device it actually used; the copies shipped here were executed on a GPU and print `device cuda`: `01` and `03` on an RTX 4080 SUPER with torch 2.7.1+cu118, `02` and `04` on an RTX 4060 Laptop GPU with torch 2.8.0+cu129.
- Training: the proposed method takes about 9 minutes on an RTX 4080 SUPER; the full ablation about 30 minutes; each baseline about 5–7 minutes.
- Checkpoints: `01_proposed/checkpoints/best_p1_proposed_seed163.pth` and `best_p2_proposed_seed163.pth` are the reference copies of the proposed weights. The ablation's `With M5` configuration *is* the proposed method and the sensitivity's default point reuses the same weights, so those checkpoints are shipped as byte-identical copies (phase 1 `0fd11d88e810b57dc68c33caa68eadb5` / 460,947 bytes, phase 2 `7b45ddb51f2909dfe3d5f92a3ac492ee` / 464,495 bytes): the package holds exactly two unique proposed checkpoints.

## Notes on how the numbers were obtained

1. **Random seed.** The proposed (`01_proposed/`), comparison (`02_comparison/`), and ablation (`03_ablation/`) experiments use a fixed random seed (`seed = 163`).
2. **Model selection.** All models keep the checkpoint with the best validation accuracy (the proposed method's phase 1 keeps the lowest validation loss). The test set is evaluated once, with that checkpoint.
3. **Hardware dependence.** The released numbers were produced on an RTX 4080 SUPER with torch 2.7.1+cu118, fp16 autocast and TF32 matmul. On the same machine and software the training runs are bit-for-bit repeatable: `retrain_check/` holds a second full run of the proposed method whose two checkpoints and test predictions are byte-identical to the released ones (see the README in that folder). On other GPUs or torch versions the trained weights, and therefore the last digits of the accuracy, can differ. Evaluating the released checkpoints needs no training, and on the machine that produced them (RTX 4080 SUPER, torch 2.7.1+cu118) the evaluation notebooks reproduce the released test counts: the proposed model exactly (61,658 of 62,002 windows, 0 windows differing from `released_results/test_predictions_proposed_seed163.csv`), all six ablation counts exactly (52,962 / 54,291 / 54,325 / 61,389 / 61,408 / 61,658; the `03` notebook shipped in this copy was re-executed on an RTX 4060 Laptop GPU (torch 2.8.0+cu129), where the six counts come out as 52,967 / 54,294 / 54,325 / 61,392 / 61,409 / 61,658, i.e. at most 5 windows (0.008 percentage points) away from the released counts), and 18 of the 20 sensitivity points exactly, the other two (rho = 0.4 and tau_1 = 0.2) being 1 window (0.0016 percentage points) off; the shipped `04` notebook was re-executed on the RTX 4060 Laptop GPU after the release, where 12 of the 20 points reproduce exactly and the other 8 differ by at most 2 windows (0.0032 percentage points). The seven baselines are computed in exact fp32 and also reproduce exactly on a different GPU (RTX 4060 Laptop GPU, torch 2.8.0+cu129): 60,813 / 61,230 / 61,269 / 61,040 / 61,010 / 60,731 / 61,317. The residual 1-window differences come from the arithmetic precision path, not from different models: the released counts were computed inside the training loop under `torch.amp.autocast("cuda")` after `torch.set_float32_matmul_precision("high")`, so re-evaluating the same checkpoints in exact fp32 shifts a count by up to 5 windows (0.008 pp), while TF32 matmul with the training batch size of 2048 - which is what the shipped notebooks use on the GPU path - puts them back within 1 window. On a CPU-only host (torch 2.7.1+cu118, exact fp32) the same checkpoints reproduce all 20 sensitivity points within 4 windows. Each notebook prints the released count next to the one it measured.
4. **DataLoader workers.** The proposed method draws its feature masks inside DataLoader workers, so `num_workers` (8) is part of the protocol; changing it changes the augmentation stream and the result.
5. **Sensitivity points.** Each point is one trained model of the proposed method evaluated once on the test set; `points.csv` lists the value, the checkpoint and the recorded test accuracy (separate validation logs for individual grid points were not archived alongside the checkpoints). The sensitivity folder releases these checkpoints for verification; the models can be retrained with `01_proposed/train_proposed.ipynb` after changing the corresponding value in that notebook's configuration block.

## Folder layout

```
├── README.md                  this file
├── requirements.txt, LICENSE, .gitignore
├── data/                      features_21d/, features_11d/, MD5SUMS.txt
├── 00_preprocessing/          01_ceemdan_decompose.py, 02_extract_features.py, 03_split_and_select_features.py, README.md
├── common/core.py             data checks, model classes and the helpers used by the evaluation notebooks
├── 01_proposed/               train_proposed.ipynb, evaluate_checkpoint.ipynb, checkpoints/, released_results/
├── 02_comparison/             train_baselines.ipynb, evaluate_checkpoints.ipynb, train_baseline.py,
│                              checkpoints/, released_results/
├── 03_ablation/               train_ablation.ipynb, evaluate_checkpoints.ipynb, checkpoints/, released_results/
├── 04_sensitivity/            evaluate_sensitivity.ipynb, points.csv, checkpoints/, fig_sensitivity.{png,pdf}
└── retrain_check/             second run of 01_proposed/, a reproducibility check (see its README.md)
```
