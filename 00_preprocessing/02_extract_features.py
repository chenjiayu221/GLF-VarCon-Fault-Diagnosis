"""Step 2: Extract 11 time- and frequency-domain features per component (7 x 11 = 77).

Reads decomposed CSV files produced by Step 1 (supporting both compact multi-window
`batch_*.csv` files and per-component `*_win<N>_<comp>.csv` files), extracts 8
time-domain and 3 FFT spectral amplitude features for each of `original` and
`imf1`..`imf6`, and exports `all_merged_features.csv`.
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DECOMPOSED_DIR = SCRIPT_DIR / "work" / "decomposed"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "work" / "all_merged_features.csv"

COMPONENTS = ["original", "imf1", "imf2", "imf3", "imf4", "imf5", "imf6"]
FEATURE_NAMES = [
    "mean",
    "std",
    "max",
    "min",
    "peak2peak",
    "rms",
    "skewness",
    "kurtosis",
    "fft_mean_amp",
    "fft_max_amp",
    "fft_var_amp",
]

BATCH_FILE_RE = re.compile(r"^batch_.+_win(\d+)-(\d+)\.csv$")
COMP_FILE_RE = re.compile(r"^(.+)_win(\d+)_(original|imf\d+)\.csv$")

LABEL_ALIASES: Dict[str, str] = {
    "aging": "Aging",
    "fault": "Fault",
    "noise": "Noise",
    "interference": "Noise",
    "normal": "Normal",
}


def canonical_label(folder_name: str) -> str:
    """Normalize folder names (e.g., `Agingone`, `Noise`) to canonical class labels."""
    stripped = re.sub(r"one$", "", folder_name.strip(), flags=re.IGNORECASE)
    return LABEL_ALIASES.get(stripped.lower(), stripped)


def parse_signal_array(data_str: str, target_length: int) -> Optional[np.ndarray]:
    """Parse JSON array text into a 1D float32 array of `target_length`."""
    try:
        cleaned = data_str.strip().replace("'", '"').replace("\n", "").replace("\t", "")
        arr = np.asarray(json.loads(cleaned), dtype=np.float32)
        return arr if arr.ndim == 1 and len(arr) == target_length else None
    except Exception as exc:
        logging.warning("Failed to parse signal vector: %s", str(exc)[:80])
        return None


def extract_component_batch_features(signals: np.ndarray, xp) -> Dict[str, np.ndarray]:
    """Compute the 11 statistical and spectral features across a batch `(B, T)`."""
    arr = xp.asarray(signals, dtype=xp.float32)
    mean_v = xp.mean(arr, axis=1)
    std_v = xp.std(arr, axis=1)
    max_v = xp.max(arr, axis=1)
    min_v = xp.min(arr, axis=1)
    p2p_v = max_v - min_v
    rms_v = xp.sqrt(xp.mean(xp.square(arr), axis=1))

    skew_v = xp.zeros_like(mean_v)
    kurt_v = xp.zeros_like(mean_v)
    valid = std_v > 1e-6
    if xp.any(valid):
        norm = (arr[valid] - mean_v[valid][:, None]) / std_v[valid][:, None]
        skew_v[valid] = xp.mean(norm ** 3, axis=1)
        kurt_v[valid] = xp.mean(norm ** 4, axis=1) - 3.0

    fft_amp = xp.abs(xp.fft.rfft(arr, axis=1))
    fft_mean = xp.mean(fft_amp, axis=1)
    fft_max = xp.max(fft_amp, axis=1)
    fft_var = xp.var(fft_amp, axis=1)

    metrics = {
        "mean": mean_v,
        "std": std_v,
        "max": max_v,
        "min": min_v,
        "peak2peak": p2p_v,
        "rms": rms_v,
        "skewness": skew_v,
        "kurtosis": kurt_v,
        "fft_mean_amp": fft_mean,
        "fft_max_amp": fft_max,
        "fft_var_amp": fft_var,
    }
    if xp is not np:
        return {k: v.get() for k, v in metrics.items()}
    return {k: np.asarray(v, dtype=np.float64) for k, v in metrics.items()}


def iter_window_records(decomposed_dir: Path, target_length: int) -> Iterator[Tuple[Dict, Dict[str, np.ndarray]]]:
    """Yield `(metadata_dict, comp_signals_dict)` across batch and per-window directories."""
    subdirs = sorted(p for p in decomposed_dir.iterdir() if p.is_dir())
    if not subdirs:
        raise ValueError(f"No class subdirectories found in {decomposed_dir}")

    for folder in subdirs:
        label = canonical_label(folder.name)
        is_with_one = "one" in folder.name.lower()
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")

        # Mode A: compact multi-window batch CSVs
        batch_files = [p for p in files if BATCH_FILE_RE.match(p.name)]
        for bf in batch_files:
            df = pd.read_csv(bf, usecols=["window_number", "type", "data"], dtype={"type": str, "data": str})
            for win_num, grp in df.groupby("window_number", sort=True):
                comp_signals: Dict[str, np.ndarray] = {}
                for _, row in grp.iterrows():
                    comp_name = str(row["type"]).strip()
                    if comp_name not in COMPONENTS:
                        continue
                    arr = parse_signal_array(str(row["data"]), target_length)
                    if arr is not None:
                        comp_signals[comp_name] = arr
                if len(comp_signals) == len(COMPONENTS):
                    meta = {
                        "sample_id": f"{'with_one' if is_with_one else 'no_one'}_{label}_{bf.name}_win{int(win_num)}",
                        "label": label,
                        "window_number": int(win_num),
                        "source_folder": folder.name,
                        "source_sample": bf.name,
                    }
                    yield meta, comp_signals

        # Mode B: per-window, per-component CSV files
        grouped: Dict[Tuple[str, int], Dict[str, Path]] = {}
        for cf in files:
            match = COMP_FILE_RE.match(cf.name)
            if not match:
                continue
            orig_stem, win_idx_str, comp_name = match.group(1), match.group(2), match.group(3)
            if comp_name in COMPONENTS:
                grouped.setdefault((orig_stem, int(win_idx_str)), {})[comp_name] = cf

        for (orig_stem, win_num), comp_paths in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
            if not all(c in comp_paths for c in COMPONENTS):
                continue
            comp_signals = {}
            valid = True
            for comp_name in COMPONENTS:
                df_c = pd.read_csv(comp_paths[comp_name], usecols=["data"], dtype={"data": str})
                if df_c.empty:
                    valid = False
                    break
                arr = parse_signal_array(str(df_c["data"].iloc[0]), target_length)
                if arr is None:
                    valid = False
                    break
                comp_signals[comp_name] = arr
            if valid:
                meta = {
                    "sample_id": f"{'with_one' if is_with_one else 'no_one'}_{label}_{orig_stem}_win{win_num}",
                    "label": label,
                    "window_number": win_num,
                    "source_folder": folder.name,
                    "source_sample": orig_stem,
                }
                yield meta, comp_signals


def flush_feature_batch(
    metas: List[Dict],
    signal_buffers: Dict[str, List[np.ndarray]],
    xp,
) -> List[Dict]:
    """Compute features for a buffered batch of windows and assemble output rows."""
    comp_stats = {
        comp: extract_component_batch_features(np.stack(signal_buffers[comp], axis=0), xp)
        for comp in COMPONENTS
    }
    out_rows: List[Dict] = []
    for idx, meta in enumerate(metas):
        row = dict(meta)
        for comp in COMPONENTS:
            stats = comp_stats[comp]
            for fname in FEATURE_NAMES:
                row[f"{comp}_{fname}"] = float(stats[fname][idx])
        out_rows.append(row)
    return out_rows


def resolve_array_backend(use_gpu: bool):
    """Return CuPy module when requested and available, otherwise NumPy."""
    if not use_gpu:
        return np
    try:
        import cupy as cp  # type: ignore

        if cp.cuda.is_available():
            logging.info("Using CuPy CUDA backend for vectorized FFT and moment computation")
            return cp
    except Exception:
        pass
    logging.info("CuPy unavailable; falling back to vectorized NumPy CPU backend")
    return np


def extract_all_features(args: argparse.Namespace) -> Path:
    """Extract all 7 x 11 features from `args.decomposed_dir` into `args.output_csv`."""
    decomposed_dir = Path(args.decomposed_dir)
    output_csv = Path(args.output_csv)
    if not decomposed_dir.is_dir():
        raise FileNotFoundError(f"Decomposed signal directory not found: {decomposed_dir}")

    xp = resolve_array_backend(args.use_gpu)
    all_rows: List[Dict] = []
    meta_buf: List[Dict] = []
    sig_buf: Dict[str, List[np.ndarray]] = {c: [] for c in COMPONENTS}

    for meta, comp_signals in tqdm(
        iter_window_records(decomposed_dir, args.target_length),
        desc="Extracting features",
    ):
        meta_buf.append(meta)
        for comp in COMPONENTS:
            sig_buf[comp].append(comp_signals[comp])

        if len(meta_buf) >= args.batch_size:
            all_rows.extend(flush_feature_batch(meta_buf, sig_buf, xp))
            meta_buf.clear()
            for comp in COMPONENTS:
                sig_buf[comp].clear()

    if meta_buf:
        all_rows.extend(flush_feature_batch(meta_buf, sig_buf, xp))

    if not all_rows:
        raise RuntimeError(f"No valid 7-component windows found under {decomposed_dir}")

    ordered_cols = ["sample_id", "label", "window_number", "source_folder", "source_sample"] + [
        f"{comp}_{fname}" for comp in COMPONENTS for fname in FEATURE_NAMES
    ]
    df_out = pd.DataFrame(all_rows)[ordered_cols]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logging.info("Saved %d rows x %d columns to %s", len(df_out), len(df_out.columns), output_csv)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 7x11 time-frequency features from decomposed signals")
    parser.add_argument("--decomposed-dir", type=Path, default=DEFAULT_DECOMPOSED_DIR, help="Input directory from step 01")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Merged 77-feature CSV output path")
    parser.add_argument("--target-length", type=int, default=2500, help="Expected samples per window")
    parser.add_argument("--batch-size", type=int, default=128, help="Vectorized batch size")
    parser.add_argument("--use-gpu", action="store_true", help="Use CuPy CUDA if installed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    t0 = time.time()
    extract_all_features(args)
    logging.info("Feature extraction finished in %.2fs", time.time() - t0)


if __name__ == "__main__":
    main()
