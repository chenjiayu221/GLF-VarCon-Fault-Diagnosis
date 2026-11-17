"""Step 1: Sliding-window CEEMDAN decomposition of raw oscilloscope signals.

Reads two-column headerless raw CSV recordings (`timestamp, voltage`) from
`--raw-dir` (default `./work/raw`), segments each recording with a sliding window
(`win_len=2500`, `step=1`), and decomposes every window into the original signal
plus 6 IMF components (`imf1`..`imf6`) via `PyEMD.CEEMDAN`.

Requires the optional `EMD-signal` package (`pip install EMD-signal`).
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = SCRIPT_DIR / "work" / "raw"
DEFAULT_OUT_DIR = SCRIPT_DIR / "work" / "decomposed"

RAW_CLASS_MAP: Dict[str, str] = {
    "aging": "Aging",
    "fault": "Fault",
    "noise": "Noise",
    "interference": "Noise",
    "normal": "Normal",
}


def resolve_class_name(filename: str) -> str:
    """Map raw CSV filename stem to canonical dataset class label."""
    stem = Path(filename).stem.lower()
    if stem in RAW_CLASS_MAP:
        return RAW_CLASS_MAP[stem]
    return Path(filename).stem.capitalize()


def make_windows(
    signal: np.ndarray,
    win_len: int,
    step: int,
    start_read: int,
    max_windows: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """Slice a 1D signal into sliding windows with physical window numbers."""
    n = len(signal)
    if n < win_len:
        raise ValueError(f"Signal length {n} is smaller than window length {win_len}")
    local_indices = np.arange(0, n - win_len + 1, step, dtype=np.int64)
    if max_windows is not None and max_windows > 0:
        local_indices = local_indices[:max_windows]
    original_starts = [int(idx) + start_read for idx in local_indices]
    window_numbers = [start // step for start in original_starts]
    segments = [signal[int(idx):int(idx) + win_len] for idx in local_indices]
    return segments, original_starts, window_numbers


def ceemdan_decompose(
    signal: np.ndarray,
    num_imfs: int = 6,
    trials: int = 100,
    epsilon: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Decompose a single window into `[original, imf1, ..., imf_K]`."""
    try:
        from PyEMD import CEEMDAN
    except ImportError as exc:
        raise ImportError(
            "PyEMD (package 'EMD-signal') is required for CEEMDAN decomposition. "
            "Install it with: pip install EMD-signal"
        ) from exc

    kwargs = {"seed": seed} if seed is not None else {}
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, **kwargs)
    imfs_raw = ceemdan(signal.astype(np.float64))
    imfs = [np.asarray(imf, dtype=np.float32) for imf in imfs_raw]

    if len(imfs) > num_imfs:
        imfs = imfs[:num_imfs]
    elif len(imfs) < num_imfs:
        pad = [np.zeros_like(signal, dtype=np.float32) for _ in range(num_imfs - len(imfs))]
        imfs.extend(pad)

    rows = [signal.astype(np.float32).reshape(1, -1)] + [imf.reshape(1, -1) for imf in imfs]
    return np.vstack(rows)


def build_window_rows(
    components: np.ndarray,
    label_id: int,
    file_index: int,
    window_number: int,
    window_original_start: int,
    global_start_read: int,
    global_end_read: int,
    win_len: int,
) -> List[Dict]:
    """Convert `[original, imf1..K]` array into output CSV record dicts."""
    window_original_end = window_original_start + win_len - 1
    rows: List[Dict] = []
    comp_names = ["original"] + [f"imf{i}" for i in range(1, components.shape[0])]
    for comp_name, arr in zip(comp_names, components):
        rows.append({
            "label": label_id,
            "file_index": file_index,
            "window_number": window_number,
            "window_original_start": window_original_start,
            "window_original_end": window_original_end,
            "global_start_read": global_start_read,
            "global_end_read": global_end_read,
            "type": comp_name,
            "data": json.dumps(arr.tolist()),
        })
    return rows


def flush_batch_csv(
    rows: List[Dict],
    class_dir: Path,
    class_name: str,
    win_start: int,
    win_end: int,
) -> None:
    """Write a batch of decomposed windows to `batch_<Class>_win<start>-<end>.csv`."""
    class_dir.mkdir(parents=True, exist_ok=True)
    out_path = class_dir / f"batch_{class_name}_win{win_start}-{win_end}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)


def flush_per_window_csvs(rows: List[Dict], class_dir: Path, class_name: str, win_num: int) -> None:
    """Write one file per component (`<Class>_win<N>_<comp>.csv`)."""
    class_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        comp_name = row["type"]
        out_path = class_dir / f"{class_name}_win{win_num}_{comp_name}.csv"
        pd.DataFrame([row]).to_csv(out_path, index=False)


def process_raw_directory(args: argparse.Namespace) -> Tuple[int, int]:
    """Decompose all raw CSV files under `args.raw_dir`."""
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    csv_files = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not csv_files:
        raise ValueError(f"No CSV files found in {raw_dir}")

    total_windows = 0
    logging.info("Found %d raw CSV files in %s", len(csv_files), raw_dir)

    for file_idx, csv_path in enumerate(csv_files):
        class_name = resolve_class_name(csv_path.name)
        class_dir = out_dir / class_name

        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] < 2:
            raise ValueError(f"{csv_path.name} has {df.shape[1]} columns; expected >= 2")

        total_length = len(df)
        file_end = total_length if args.end_read is None else min(args.end_read, total_length)
        if args.start_read < 0 or file_end <= args.start_read:
            raise ValueError(
                f"Invalid slice [{args.start_read}, {file_end}) for {csv_path.name} (length={total_length})"
            )

        signal = df.iloc[args.start_read:file_end, 1].values.astype(np.float32)
        segments, orig_starts, win_nums = make_windows(
            signal=signal,
            win_len=args.win_len,
            step=args.step,
            start_read=args.start_read,
            max_windows=args.max_windows,
        )
        logging.info(
            "[%s] %s: %d windows (win %d..%d) from slice [%d, %d)",
            class_name,
            csv_path.name,
            len(segments),
            win_nums[0],
            win_nums[-1],
            args.start_read,
            file_end,
        )

        batch_rows: List[Dict] = []
        batch_first_win: Optional[int] = None

        for seg, orig_start, win_num in tqdm(
            zip(segments, orig_starts, win_nums),
            total=len(segments),
            desc=class_name,
            leave=False,
        ):
            components = ceemdan_decompose(
                seg,
                num_imfs=args.num_imfs,
                trials=args.trials,
                epsilon=args.epsilon,
                seed=args.seed,
            )
            win_rows = build_window_rows(
                components=components,
                label_id=file_idx,
                file_index=file_idx,
                window_number=win_num,
                window_original_start=orig_start,
                global_start_read=args.start_read,
                global_end_read=file_end,
                win_len=args.win_len,
            )
            total_windows += 1

            if args.format == "per-window":
                flush_per_window_csvs(win_rows, class_dir, class_name, win_num)
            else:
                if batch_first_win is None:
                    batch_first_win = win_num
                batch_rows.extend(win_rows)
                if len(batch_rows) // (args.num_imfs + 1) >= args.batch_windows:
                    flush_batch_csv(batch_rows, class_dir, class_name, batch_first_win, win_num)
                    batch_rows = []
                    batch_first_win = None

        if args.format == "batch" and batch_rows and batch_first_win is not None:
            last_win = int(batch_rows[-1]["window_number"])
            flush_batch_csv(batch_rows, class_dir, class_name, batch_first_win, last_win)

    return total_windows, args.num_imfs + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding-window CEEMDAN decomposition")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with raw CSV files")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for decomposed signals")
    parser.add_argument("--num-imfs", type=int, default=6, help="Number of IMF components to retain per window")
    parser.add_argument("--win-len", type=int, default=2500, help="Sliding window length in samples")
    parser.add_argument("--step", type=int, default=1, help="Sliding window step size")
    parser.add_argument("--start-read", type=int, default=0, help="Start sample index in raw recording")
    parser.add_argument("--end-read", type=int, default=None, help="End sample index (default: full length)")
    parser.add_argument("--max-windows", type=int, default=None, help="Optional window cap per file (for quick tests)")
    parser.add_argument("--trials", type=int, default=100, help="CEEMDAN ensemble realizations (PyEMD trials)")
    parser.add_argument("--epsilon", type=float, default=0.005, help="CEEMDAN white-noise amplitude scaling")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for PyEMD CEEMDAN")
    parser.add_argument(
        "--format",
        choices=["batch", "per-window"],
        default="batch",
        help="Output file layout: compact multi-window batch CSVs or per-component CSVs",
    )
    parser.add_argument("--batch-windows", type=int, default=10, help="Windows per CSV in batch mode")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    t0 = time.time()
    total_windows, comps_per_win = process_raw_directory(args)
    logging.info(
        "Completed CEEMDAN decomposition in %.2fs: windows=%d, components_per_window=%d, out=%s",
        time.time() - t0,
        total_windows,
        comps_per_win,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
