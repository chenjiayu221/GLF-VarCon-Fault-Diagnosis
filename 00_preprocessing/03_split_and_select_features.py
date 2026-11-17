"""Step 3: Temporal buffer splitting, training-only standardization, and intra-group SVM-RFE.

Reads the 77-feature table produced by Step 2, applies the chronological
non-overlapping split `[Test Head 10%] - [Gap] - [Train] - [Gap] - [Val] - [Gap] - [Test Tail 10%]`
per class, fits median imputation and Z-score standardization exclusively on
the training split, and exports both:
  1) `features_11d/selected_features_{train,val,test}.csv` (11 raw-waveform features)
  2) `features_21d/selected_features_{train,val,test}.csv` (3 SVM-RFE features per group x 7 groups)
"""
import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "work" / "all_merged_features.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "work" / "processed_data"
RELEASED_DATA_DIR = REPO_ROOT / "data"

META_COLUMNS = ["sample_id", "label", "window_number", "source_folder", "source_sample"]
COMPONENTS = ["original", "imf1", "imf2", "imf3", "imf4", "imf5", "imf6"]
FEATS_PER_GROUP = 11


def split_with_temporal_buffer(
    df: pd.DataFrame,
    test_head_ratio: float = 0.1,
    val_ratio: float = 0.1,
    test_tail_ratio: float = 0.1,
    gap_size: int = 2500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split each class with temporal guard bands between partitions."""
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for label in df["label"].unique():
        label_df = df[df["label"] == label].sort_values(by="window_number")
        sorted_indices = label_df.index.tolist()
        n = len(sorted_indices)

        idx_p1 = int(n * test_head_ratio)
        idx_p2 = int(n * (1.0 - test_tail_ratio - val_ratio))
        idx_p3 = int(n * (1.0 - test_tail_ratio))

        test_head = sorted_indices[:idx_p1]
        train_start = idx_p1 + gap_size
        train_end = idx_p2 - gap_size
        val_start = idx_p2
        val_end = idx_p3 - gap_size
        test_tail = sorted_indices[idx_p3:]

        if train_start >= train_end or val_start >= val_end or not test_head or not test_tail:
            raise ValueError(
                f"Class '{label}' has too few windows ({n}) for ratios and gap_size={gap_size}"
            )

        train_part = sorted_indices[train_start:train_end]
        val_part = sorted_indices[val_start:val_end]

        train_idx.extend(train_part)
        val_idx.extend(val_part)
        test_idx.extend(test_head)
        test_idx.extend(test_tail)

        dropped = (train_start - idx_p1) + (val_start - train_end) + (idx_p3 - val_end)
        logging.info(
            "Class %-8s | total=%d train=%d val=%d test=%d (buffer dropped=%d)",
            label,
            n,
            len(train_part),
            len(val_part),
            len(test_head) + len(test_tail),
            dropped,
        )

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def export_split_csvs(
    splits: Dict[str, pd.DataFrame],
    arrays: Dict[str, np.ndarray],
    feature_cols: List[str],
    out_dir: Path,
) -> None:
    """Write `selected_features_{train,val,test}.csv` with meta + feature columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        meta_cols = [c for c in META_COLUMNS if c in splits[split_name].columns]
        meta_sub = splits[split_name][meta_cols].reset_index(drop=True)
        feat_sub = pd.DataFrame(arrays[split_name], columns=feature_cols)
        out_path = out_dir / f"selected_features_{split_name}.csv"
        pd.concat([meta_sub, feat_sub], axis=1).to_csv(out_path, index=False, encoding="utf-8-sig")
        logging.info("Wrote %s (%d rows x %d cols)", out_path, len(meta_sub), len(meta_cols) + len(feature_cols))


def run_selection(args: argparse.Namespace) -> None:
    """Execute buffer splitting, training-only scaling, and intra-group SVM-RFE."""
    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir).resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Merged feature CSV not found: {input_csv}")

    if not args.allow_overwrite_released and out_dir == RELEASED_DATA_DIR.resolve():
        raise PermissionError(
            f"Refusing to write directly into released data dir {out_dir}; "
            "pass a working --out-dir or --allow-overwrite-released."
        )

    df = pd.read_csv(input_csv)
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    expected_dim = len(COMPONENTS) * FEATS_PER_GROUP
    if len(feature_cols) != expected_dim:
        raise ValueError(f"Expected {expected_dim} features (7 x 11), found {len(feature_cols)}")

    train_idx, val_idx, test_idx = split_with_temporal_buffer(
        df=df,
        test_head_ratio=args.test_head_ratio,
        val_ratio=args.val_ratio,
        test_tail_ratio=args.test_tail_ratio,
        gap_size=args.gap_size,
    )
    splits = {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }

    # Imputation and standardization fitted exclusively on the training split
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(splits["train"][feature_cols].values)
    X_val_imp = imputer.transform(splits["val"][feature_cols].values)
    X_test_imp = imputer.transform(splits["test"][feature_cols].values)

    scaler = StandardScaler()
    X_scaled = {
        "train": scaler.fit_transform(X_train_imp),
        "val": scaler.transform(X_val_imp),
        "test": scaler.transform(X_test_imp),
    }
    y_train = splits["train"]["label"].values

    # Export 11D raw-waveform scaled features (group 0)
    orig_cols = feature_cols[:FEATS_PER_GROUP]
    orig_arrays = {k: v[:, :FEATS_PER_GROUP] for k, v in X_scaled.items()}
    export_split_csvs(splits, orig_arrays, orig_cols, out_dir / "features_11d")

    # Intra-group linear-SVM RFE fitted exclusively on the training split (7 x keep_per_group)
    selected_blocks: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    selected_cols: List[str] = []
    rfe_selectors: Dict[str, RFE] = {}

    for g_idx, comp in enumerate(COMPONENTS):
        s_col = g_idx * FEATS_PER_GROUP
        e_col = s_col + FEATS_PER_GROUP
        group_cols = feature_cols[s_col:e_col]

        selector = RFE(
            estimator=SVC(kernel="linear"),
            n_features_to_select=args.keep_per_group,
            step=1,
        )
        selector.fit(X_scaled["train"][:, s_col:e_col], y_train)
        mask = selector.get_support()
        chosen = [group_cols[i] for i in range(FEATS_PER_GROUP) if mask[i]]
        selected_cols.extend(chosen)
        rfe_selectors[comp] = selector

        for split_name in ("train", "val", "test"):
            selected_blocks[split_name].append(X_scaled[split_name][:, s_col:e_col][:, mask])
        logging.info("Group %-8s selected (%d): %s", comp, len(chosen), chosen)

    final_arrays = {split_name: np.hstack(selected_blocks[split_name]) for split_name in ("train", "val", "test")}
    dir_21d = out_dir / f"features_{len(COMPONENTS) * args.keep_per_group}d"
    export_split_csvs(splits, final_arrays, selected_cols, dir_21d)

    pkg_path = dir_21d / "feature_selector_package.pkl"
    with open(pkg_path, "wb") as f:
        pickle.dump(
            {
                "imputer": imputer,
                "scaler": scaler,
                "rfe_models_zoo": rfe_selectors,
                "final_features_list": selected_cols,
            },
            f,
        )
    logging.info("Saved preprocessing package to %s", pkg_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal buffer split + intra-group SVM-RFE selection")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="77-feature CSV from step 02")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output root for 11d/21d splits")
    parser.add_argument("--keep-per-group", type=int, default=3, help="Features retained per component group (7 x 3 = 21)")
    parser.add_argument("--test-head-ratio", type=float, default=0.1, help="Head test split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-tail-ratio", type=float, default=0.1, help="Tail test split ratio")
    parser.add_argument("--gap-size", type=int, default=2500, help="Guard band windows removed at split boundaries")
    parser.add_argument(
        "--allow-overwrite-released",
        action="store_true",
        help="Required if targeting the repository's released data/ directory",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_selection(args)


if __name__ == "__main__":
    main()
