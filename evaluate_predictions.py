"""
evaluate_predictions.py — Standalone Benchmark Prediction Evaluation
===================================================================
Protocol: myops380_voxel_v3 (Exact SCAR Parity)
Evaluates test patient volumes across 5 target regions:
  1. Myocardium (MYO / myocardial_ring): IDs [1, 2, 3]
  2. Edema (EDEMA / edema): ID [2] (exclusive)
  3. Scar (SCAR / scar): ID [3]
  4. LGE-abnormal (LGE-abn / edema_inclusive): IDs [2, 3]
  5. Normal Myocardium (NORM / normal_myocardium): ID [1]

Outputs:
  - <output_dir>/per_case.csv          (patient rows + Mean, Std, Median)
  - <output_dir>/per_case_detailed.csv (380-row long format, exact SCAR schema)
  - <output_dir>/metrics.json          (benchmark summary metrics & counts)
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


# Canonical benchmark protocol descriptor
BENCHMARK_PROTOCOL = {
    "id": "myops380_voxel_v3",
    "distance_unit": "voxel",
    "voxelspacing": None,
    "axis_order": "HWD",
    "regions": {
        "normal_myocardium": [1],
        "edema": [2],
        "scar": [3],
        "edema_inclusive": [2, 3],
        "myocardial_ring": [1, 2, 3],
    },
    "region_aliases": {
        "NORM": "normal_myocardium",
        "EDEMA": "edema",
        "SCAR": "scar",
        "LGE-abn": "edema_inclusive",
        "MYO": "myocardial_ring",
    },
    "primary_empty_policy": "both_empty: excluded; one_empty: Dice/IoU=0, distances undefined",
    "aggregation": "equal patient weight within each region; exclude undefined values and report counts",
    "selection_metric": "avg_pathology_inclusive_dice",
    "pathology_regions": ["scar", "edema"],
    "pathology_aggregation": "arithmetic mean of scar and exclusive-edema patient means; both required",
}

CANONICAL_LUT = np.array([0, 1, 3, 2], dtype=np.uint8)
EXPECTED_TEST_CASES = 76


def setup_logger(output_dir: Path) -> logging.Logger:
    """Configures console and file logger for evaluation."""
    logger = logging.getLogger("evaluate_predictions")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(output_dir / "evaluation.log", mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Optional[Sequence[float]] = None,
    compute_distance: bool = True,
    empty_mode: str = "excluded",
) -> Dict[str, Any]:
    """Computes Dice, IoU, HD95, and ASSD with strict empty-mask policy matching SCAR."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)

    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs target {truth.shape}")

    if spacing is not None:
        spacing = np.asarray(spacing, dtype=float)
        if spacing.shape != (pred.ndim,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Spacing must contain one finite positive value per array axis.")

    n_pred, n_true = int(pred.sum()), int(truth.sum())

    # Corner Case 1: Both empty
    if n_pred == 0 and n_true == 0:
        if empty_mode == "defined":
            return {
                "dice": 1.0,
                "iou": 1.0,
                "hd95": 0.0,
                "asd": 0.0,
                "status": "both_empty",
                "precision": 1.0,
                "recall": 1.0,
            }
        return {
            "dice": None,
            "iou": None,
            "hd95": None,
            "asd": None,
            "status": "both_empty",
            "precision": None,
            "recall": None,
        }

    overlap = int(np.logical_and(pred, truth).sum())
    result: Dict[str, Any] = {
        "dice": float(2.0 * overlap / (n_pred + n_true)),
        "iou": float(overlap / (n_pred + n_true - overlap)),
        "hd95": None,
        "asd": None,
        "status": "ok",
        "precision": float(overlap / n_pred) if n_pred > 0 else (1.0 if empty_mode == "defined" else None),
        "recall": float(overlap / n_true) if n_true > 0 else (1.0 if empty_mode == "defined" else None),
    }

    # Corner Case 2 & 3: One empty
    if n_pred == 0 or n_true == 0:
        if empty_mode == "defined":
            result["hd95"] = float("inf")
            result["asd"] = float("inf")
            result["status"] = "one_empty"
        else:
            result["status"] = "prediction_empty" if n_pred == 0 else "target_empty"
        return result

    if not compute_distance:
        return result

    # Surface distance calculation using 1-connected structuring element
    structure = generate_binary_structure(pred.ndim, 1)
    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    true_surface = truth ^ binary_erosion(truth, structure=structure, border_value=0)

    dt_true = distance_transform_edt(~true_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    distances = np.concatenate((dt_true[pred_surface], dt_pred[true_surface]))

    if distances.size > 0:
        result["hd95"] = float(np.percentile(distances, 95))
        result["asd"] = float(np.mean(distances))

    return result


def evaluate_single_volume(
    prediction: np.ndarray,
    target_raw: np.ndarray,
    case_id: str,
    spacing: Optional[Sequence[float]] = None,
    compute_distance: bool = True,
    empty_mode: str = "excluded",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluates one patient volume across all 5 regions."""
    # Convert GT from legacy order to canonical order
    target = CANONICAL_LUT[target_raw.astype(int)]

    detailed_rows = []
    case_summary: Dict[str, Any] = {"case_id": case_id}
    distance_unit = "mm" if spacing is not None else "voxel"

    for reg_name, ids in BENCHMARK_PROTOCOL["regions"].items():
        pred_mask = np.isin(prediction, ids)
        true_mask = np.isin(target, ids)

        m = binary_metrics(
            pred_mask,
            true_mask,
            spacing=spacing,
            compute_distance=compute_distance,
            empty_mode=empty_mode,
        )
        n_pred = int(pred_mask.sum())
        n_true = int(true_mask.sum())

        row = {
            "case": case_id,
            "region": reg_name,
            "dice": m["dice"],
            "iou": m["iou"],
            "hd95": m["hd95"],
            "asd": m["asd"],
            "status": m["status"],
            "precision": m["precision"],
            "recall": m["recall"],
            "prediction_voxels": n_pred,
            "target_voxels": n_true,
            "hd95_unit": distance_unit,
            f"hd95_{distance_unit}": m["hd95"],
            f"asd_{distance_unit}": m["asd"],
        }
        detailed_rows.append(row)

        # Standard canonical columns in wide CSV
        case_summary[f"{reg_name}_status"] = m["status"]
        case_summary[f"{reg_name}_dice"] = m["dice"]
        case_summary[f"{reg_name}_iou"] = m["iou"]
        case_summary[f"{reg_name}_hd95"] = m["hd95"]
        case_summary[f"{reg_name}_assd"] = m["asd"]

        # Also store short alias columns for compatibility
        aliases = [k for k, v in BENCHMARK_PROTOCOL["region_aliases"].items() if v == reg_name]
        if aliases:
            alias = aliases[0].lower().replace("-", "_")
            if alias != reg_name:
                case_summary[f"{alias}_status"] = m["status"]
                case_summary[f"{alias}_dice"] = m["dice"]
                case_summary[f"{alias}_iou"] = m["iou"]
                case_summary[f"{alias}_hd95"] = m["hd95"]
                case_summary[f"{alias}_assd"] = m["asd"]

    # Pathology composite metrics for wide CSV
    scar_dice = case_summary.get("scar_dice")
    edema_dice = case_summary.get("edema_dice")
    lge_abn_dice = case_summary.get("edema_inclusive_dice")

    case_summary["avg_pathology_dice"] = (
        float(np.mean([scar_dice, edema_dice]))
        if (scar_dice is not None and edema_dice is not None)
        else None
    )
    case_summary["avg_pathology_inclusive_dice"] = (
        float(np.mean([scar_dice, lge_abn_dice]))
        if (scar_dice is not None and lge_abn_dice is not None)
        else None
    )

    return detailed_rows, case_summary


def compute_cohort_summary(
    detailed_rows: List[Dict[str, Any]],
    case_summaries: List[Dict[str, Any]],
    distance_unit: str = "voxel",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Computes cohort summary metrics for metrics.json and CSV summary rows."""
    summary: Dict[str, Any] = {}
    regions = list(BENCHMARK_PROTOCOL["regions"].keys())

    for name in regions:
        subset = [r for r in detailed_rows if r["region"] == name]
        region_stats: Dict[str, Any] = {
            "cases": len(subset),
            "status_counts": dict(Counter(r["status"] for r in subset)),
        }
        for metric in ("dice", "iou", "precision", "recall", f"hd95_{distance_unit}", f"asd_{distance_unit}"):
            vals = [r[metric] for r in subset if r[metric] is not None]
            region_stats[f"mean_{metric}"] = float(np.mean(vals)) if vals else None
            region_stats[f"std_{metric}"] = float(np.std(vals)) if vals else None
            region_stats[f"median_{metric}"] = float(np.median(vals)) if vals else None
            region_stats[f"{metric}_defined_cases"] = len(vals)
            region_stats[f"{metric}_undefined_cases"] = len(subset) - len(vals)

        # Convenience aliases for standard metrics
        region_stats["mean_hd95_voxel"] = region_stats.get(f"mean_hd95_{distance_unit}")
        region_stats["mean_asd_voxel"] = region_stats.get(f"mean_asd_{distance_unit}")
        summary[name] = region_stats

    # Pathology composite means across regions
    for metric in ("dice", "iou", f"hd95_{distance_unit}", f"asd_{distance_unit}"):
        v_excl = [summary[r][f"mean_{metric}"] for r in ("scar", "edema")]
        summary[f"avg_pathology_{metric}"] = float(np.mean(v_excl)) if all(v is not None for v in v_excl) else None

        v_incl = [summary[r][f"mean_{metric}"] for r in ("scar", "edema_inclusive")]
        summary[f"avg_pathology_inclusive_{metric}"] = float(np.mean(v_incl)) if all(v is not None for v in v_incl) else None

    # Summary rows for wide CSV (Mean, Std, Median)
    summary_rows = []
    for agg_name, agg_func in (("Mean", np.mean), ("Std", np.std), ("Median", np.median)):
        agg_row: Dict[str, Any] = {"case_id": agg_name}
        for k in case_summaries[0].keys():
            if k == "case_id":
                continue
            if k.endswith("_status"):
                agg_row[k] = ""
            else:
                vals = [cs[k] for cs in case_summaries if cs.get(k) is not None]
                agg_row[k] = float(agg_func(vals)) if vals else ""
        summary_rows.append(agg_row)

    return summary, summary_rows


def load_prediction_volume(pred_dir: Path, case_id: str) -> np.ndarray:
    """Finds and loads the 3D uint8 canonical prediction array."""
    candidates = [
        pred_dir / f"{case_id}_pred.npz",
        pred_dir / f"{case_id}.npz",
        pred_dir / f"{case_id}.npy.npz",
    ]
    for p in candidates:
        if p.is_file():
            with np.load(p) as data:
                for k in ("prediction", "pred", "arr_0"):
                    if k in data:
                        return np.asarray(data[k], dtype=np.uint8)
    raise FileNotFoundError(f"No prediction file found for {case_id} in {pred_dir}")


def load_ground_truth_volume(data_root: Path, case_id: str, modality: str = "bSSFP") -> np.ndarray:
    """Loads raw ground truth label volume from H5."""
    candidates = [
        data_root / modality / "test_vol_h5" / f"{case_id}.npy.h5",
        data_root / modality / "test_vol_h5" / f"{case_id}.h5",
        data_root / "test_vol_h5" / f"{case_id}.npy.h5",
        data_root / "test_vol_h5" / f"{case_id}.h5",
    ]
    # Also check other modalities if primary modality does not contain label
    for alt_mod in ("bSSFP", "LGE", "T2w"):
        candidates.append(data_root / alt_mod / "test_vol_h5" / f"{case_id}.npy.h5")

    for p in candidates:
        if p.is_file():
            with h5py.File(p, "r") as f:
                if "label" in f:
                    return np.asarray(f["label"][:], dtype=np.float32)

    raise FileNotFoundError(f"No GT label found for {case_id} in {data_root}")


def evaluate_dataset(
    predictions_dir: Union[str, Path],
    data_root: Union[str, Path],
    split_file: Union[str, Path],
    output_dir: Union[str, Path],
    spacing: Optional[Sequence[float]] = None,
    modality: str = "bSSFP",
    compute_distance: bool = True,
    cases: Optional[Sequence[str]] = None,
    empty_mode: str = "excluded",
) -> Dict[str, Any]:
    """Evaluates all cases in split against predictions."""
    pred_dir = Path(predictions_dir).resolve()
    data_root = Path(data_root).resolve()
    split_path = Path(split_file).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)
    logger.info("Initializing prediction evaluation under protocol %s", BENCHMARK_PROTOCOL["id"])
    logger.info("Predictions dir: %s", pred_dir)
    logger.info("Data root: %s", data_root)
    logger.info("Split path: %s", split_path)
    logger.info("Output dir: %s", out_dir)

    # Resolve case IDs
    if cases is not None and len(cases) > 0:
        case_ids = list(cases)
    elif split_path.is_file():
        case_ids = [
            line.strip().replace(".npy.h5", "").replace(".h5", "")
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif split_path.is_dir():
        cand = split_path / "test_vol.txt"
        if not cand.is_file():
            cand = split_path / "test.txt"
        if cand.is_file():
            case_ids = [
                line.strip().replace(".npy.h5", "").replace(".h5", "")
                for line in cand.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            raise FileNotFoundError(f"No test_vol.txt or test.txt found in {split_path}")
    else:
        raise FileNotFoundError(f"Split file or directory not found: {split_path}")

    if len(case_ids) != EXPECTED_TEST_CASES:
        logger.warning("Expected %d test cases, found %d in %s", EXPECTED_TEST_CASES, len(case_ids), split_path)

    all_detailed_rows = []
    all_case_summaries = []
    distance_unit = "mm" if spacing is not None else "voxel"

    t0 = time.perf_counter()
    for idx, case_id in enumerate(case_ids, 1):
        pred = load_prediction_volume(pred_dir, case_id)
        gt_raw = load_ground_truth_volume(data_root, case_id, modality=modality)

        detailed_rows, case_summary = evaluate_single_volume(
            pred,
            gt_raw,
            case_id,
            spacing=spacing,
            compute_distance=compute_distance,
            empty_mode=empty_mode,
        )
        all_detailed_rows.extend(detailed_rows)
        all_case_summaries.append(case_summary)

        logger.info(
            "[%02d/%02d] %s | MYO: %.3f | Edema: %s | Scar: %s | LGE-abn: %s",
            idx,
            len(case_ids),
            case_id,
            case_summary.get("myocardial_ring_dice", case_summary.get("myo_dice", 0.0)) or 0.0,
            f"{case_summary['edema_dice']:.3f}" if case_summary.get("edema_dice") is not None else "EMPTY",
            f"{case_summary['scar_dice']:.3f}" if case_summary.get("scar_dice") is not None else "EMPTY",
            f"{case_summary['edema_inclusive_dice']:.3f}" if case_summary.get("edema_inclusive_dice") is not None else "EMPTY",
        )

    elapsed = time.perf_counter() - t0
    logger.info("Completed evaluation of %d volumes in %.2f seconds.", len(case_ids), elapsed)

    # Cohort summarization
    summary, summary_rows = compute_cohort_summary(
        all_detailed_rows,
        all_case_summaries,
        distance_unit=distance_unit,
    )

    # 1. Write per_case.csv (patient rows + Mean, Std, Median)
    per_case_csv = out_dir / "per_case.csv"
    with open(per_case_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_case_summaries[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_case_summaries)
        writer.writerows(summary_rows)
    logger.info("Saved patient-level wide CSV to %s", per_case_csv)

    # 2. Write per_case_detailed.csv (SCAR-parity long format)
    per_case_detailed_csv = out_dir / "per_case_detailed.csv"
    with open(per_case_detailed_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_detailed_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_detailed_rows)
    logger.info("Saved detailed SCAR-parity CSV to %s", per_case_detailed_csv)

    # 3. Write metrics.json
    summary["metadata"] = {
        "protocol_id": "myops380_train304_test76_last_v1",
        "metric_protocol": "myops380_voxel_v3",
        "dataset_id": "myops380",
        "case_count": len(case_ids),
        "distance_unit": distance_unit,
        "evaluation_seconds": round(elapsed, 4),
        "primary_selection_metric": BENCHMARK_PROTOCOL["selection_metric"],
        "primary_selection_value": summary.get(BENCHMARK_PROTOCOL["selection_metric"]),
    }
    summary["benchmark_protocol"] = BENCHMARK_PROTOCOL["id"]
    summary["metric_protocol"] = BENCHMARK_PROTOCOL["id"]
    summary["dataset_id"] = "myops380"
    summary["case_count"] = len(case_ids)
    summary["distance_unit"] = distance_unit

    metrics_json = out_dir / "metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved benchmark metrics summary JSON to %s", metrics_json)

    # Log summary table
    logger.info("======================= BENCHMARK SUMMARY =======================")
    logger.info("Avg Pathology Inclusive Dice (Primary Selection): %s", summary.get("avg_pathology_inclusive_dice"))
    logger.info("Avg Pathology Exclusive Dice:                   %s", summary.get("avg_pathology_dice"))
    for r in ("myocardial_ring", "edema", "scar", "edema_inclusive", "normal_myocardium"):
        st = summary[r]
        logger.info(
            "%-18s | Mean DSC: %s | Defined: %d/%d | Mean HD95: %s %s",
            r,
            f"{st['mean_dice']:.4f}" if st['mean_dice'] is not None else "None",
            st['dice_defined_cases'],
            st['cases'],
            f"{st[f'mean_hd95_{distance_unit}']:.3f}" if st[f'mean_hd95_{distance_unit}'] is not None else "None",
            distance_unit,
        )
    logger.info("================================================================")

    return summary


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for evaluate_predictions.py."""
    parser = argparse.ArgumentParser(
        description="Evaluate MyoPS380 3D test volume predictions under protocol myops380_voxel_v3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred_dir", "--pred-dir", "--predictions_dir", "--predictions-dir",
        default=None,
        dest="pred_dir",
        help="Directory containing <CaseID>_pred.npz files",
    )
    parser.add_argument(
        "--run_dir", "--run-dir",
        default=None,
        dest="run_dir",
        help="Experiment run directory (infers pred_dir and output_dir)",
    )
    parser.add_argument(
        "--data_root", "--data-root",
        default="D:/NCKH/Dataset/MyoPS380/Processed_data",
        dest="data_root",
        help="Path to Processed_data root directory",
    )
    parser.add_argument(
        "--split_dir", "--split-dir", "--split_file", "--split-file",
        default="D:/NCKH/SCAR/preprocessing/splits/myops380_no_val/test_vol.txt",
        dest="split_path",
        help="Path to test_vol.txt or directory containing test_vol.txt",
    )
    parser.add_argument(
        "--output_dir", "--output-dir",
        default=None,
        dest="output_dir",
        help="Output directory for CSV and JSON metrics (defaults to <run_dir>/evaluation)",
    )
    parser.add_argument(
        "--modality",
        default="bSSFP",
        help="Modality subfolder for GT volume labels",
    )
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=None,
        help="Optional physical voxel spacing in mm (e.g. 1.0 1.0 5.0)",
    )
    parser.add_argument(
        "--no_distance", "--no-distance",
        action="store_true",
        dest="no_distance",
        help="Skip surface distance computation for fast validation",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Explicit list of CaseIDs to evaluate (e.g. --cases case0001 case0006)",
    )
    parser.add_argument(
        "--empty_mode",
        choices=["excluded", "defined"],
        default="excluded",
        help="Empty mask handling mode: excluded (SCAR benchmark standard) or defined",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI entry point for evaluate_predictions.py."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        pred_dir = Path(args.pred_dir).resolve() if args.pred_dir else run_dir / "test_predictions"
        output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "evaluation"
    else:
        if args.pred_dir is None:
            raise ValueError("Must provide either --pred_dir or --run_dir")
        pred_dir = Path(args.pred_dir).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else pred_dir.parent / "evaluation"

    return evaluate_dataset(
        predictions_dir=pred_dir,
        data_root=args.data_root,
        split_file=args.split_path,
        output_dir=output_dir,
        spacing=args.spacing,
        modality=args.modality,
        compute_distance=not args.no_distance,
        cases=args.cases,
        empty_mode=args.empty_mode,
    )


if __name__ == "__main__":
    main()
