# Preprocessing pipeline (`00_preprocessing/`)

Reference implementation showing how raw two-column oscilloscope waveforms (`timestamp, voltage`) are converted into the standardized `11`-dimensional and `21`-dimensional feature splits. Raw recordings and full intermediate CEEMDAN window CSVs are omitted to keep the repository compact; pass your own raw CSV folder via `--raw-dir` to run this reference pipeline.

## Scripts

1. `01_ceemdan_decompose.py`
   - Reads two-column headerless CSV recordings from `--raw-dir` (default `work/raw/`).
   - Segments each waveform with sliding window length `2500` and step `1` (`start_read=0`), runs `PyEMD.CEEMDAN` (`trials=100`, `epsilon=0.005`), and retains the original window plus the first `6` IMF components (`original`, `imf1`..`imf6`).
   - Requires the optional `EMD-signal` package (`pip install EMD-signal`).

2. `02_extract_features.py`
   - Reads decomposed CSVs (`batch_*.csv` or per-window `*_win*_*.csv`) and computes `11` features per component (`mean`, `std`, `max`, `min`, `peak2peak`, `rms`, `skewness`, `kurtosis`, `fft_mean_amp`, `fft_max_amp`, `fft_var_amp`), yielding `7 x 11 = 77` features per window in `work/all_merged_features.csv`.

3. `03_split_and_select_features.py`
   - Sorts windows chronologically by `window_number` per class and splits them into `[Test Head 10%] - [Gap 2500] - [Train] - [Gap 2500] - [Val] - [Gap 2500] - [Test Tail 10%]`.
   - Fits median imputation (`SimpleImputer`) and Z-score standardization (`StandardScaler`) exclusively on the training split and applies them to `val` and `test`.
   - Exports `work/processed_data/features_11d/` (the 11 `original_*` features) and runs intra-group `RFE(SVC(kernel='linear'), n_features_to_select=3)` independently on each of the 7 component groups (fitted on `train` only) to export `work/processed_data/features_21d/` (`3 x 7 = 21` features).

## Example usage

Run from the repository root (`opennew_mlp8/`):

```bash
# Quick end-to-end smoke test (40 windows per class, reduced gap and trials)
python 00_preprocessing/01_ceemdan_decompose.py --raw-dir /path/to/raw_csvs --max-windows 40 --trials 5
python 00_preprocessing/02_extract_features.py
python 00_preprocessing/03_split_and_select_features.py --gap-size 2

# Full pipeline (compute-intensive due to window-wise CEEMDAN)
python 00_preprocessing/01_ceemdan_decompose.py --raw-dir /path/to/raw_csvs
python 00_preprocessing/02_extract_features.py
python 00_preprocessing/03_split_and_select_features.py
```

By default all outputs are written under `00_preprocessing/work/` (git-ignored) so the released checksummed splits under `../data/features_11d/` and `../data/features_21d/` are never overwritten.
