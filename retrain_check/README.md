# Reproducibility check: a second training run of the proposed method

This folder is the output of a second, independent run of
`../01_proposed/train_proposed.ipynb`, executed on 2026-11-02 on an NVIDIA RTX 4080 SUPER
(torch 2.7.1+cu118, CUDA 11.8) with the configuration the notebook ships (seed 163,
batch 2048, 100 + 50 epochs, fp16 autocast, TF32 matmul) and the same 21-dim feature files
as `../data/features_21d/` (fingerprint in `results/data_fingerprint.csv`).
`train_proposed.executed.ipynb` is that notebook with its outputs, exactly as it ran here.

The point of the run is to show that the released training code reproduces the released
weights exactly, not just approximately:

| file | this run | `../01_proposed/` | |
|---|---|---|---|
| `best_p1_proposed_seed163.pth` | `0fd11d88e810b57dc68c33caa68eadb5` | `0fd11d88e810b57dc68c33caa68eadb5` | identical |
| `best_p2_proposed_seed163.pth` | `7b45ddb51f2909dfe3d5f92a3ac492ee` | `7b45ddb51f2909dfe3d5f92a3ac492ee` | identical |
| `test_predictions_proposed_seed163.csv` | `ed423b4605f76e30a8085dfc2b8aa202` | `ed423b4605f76e30a8085dfc2b8aa202` | identical |

All metrics are identical as well: 61,658 of 62,002 test windows (99.445179%, macro-F1
0.994451, per-class F1 0.989011 / 0.988794 / 1.0 / 1.0). Only the wall-clock differs,
543.2 s here against 534.3 s in the released run.

`../01_proposed/` remains the released run - it is what the evaluation notebook loads - and
this folder is the evidence that the training is repeatable, not a second set of numbers.
