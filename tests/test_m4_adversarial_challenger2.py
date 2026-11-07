"""
tests/test_m4_adversarial_challenger2.py
========================================
Adversarial Challenger 2 Test Suite for Milestone 4 (evaluate_predictions.py).

Rigorous stress-testing of evaluation metric mathematical parity and edge cases:
1. Direct numerical equivalence against SCAR reference implementation (D:/NCKH/SCAR/training/metrics/surface_distance.py)
   on complex non-convex, hollow, thin, and disconnected 3D geometries.
2. Exhaustive empty mask matrix (both-empty, one-empty) under excluded vs defined modes.
3. Target region disjointness and hierarchical completeness across all 5 benchmark regions.
4. Label reconciliation integrity under CANONICAL_LUT = [0, 1, 3, 2] against label inversion and double-swapping attacks.
5. Mathematical integrity of CSV (per_case.csv) and JSON (metrics.json) summary statistics across a 76-patient cohort.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pytest
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_predictions as ep
from evaluate_predictions import (
    BENCHMARK_PROTOCOL,
    CANONICAL_LUT,
    binary_metrics as ep_binary_metrics,
    compute_cohort_summary,
    evaluate_dataset,
    evaluate_single_volume,
)

# Dynamically import the real, read-only SCAR reference implementation
SCAR_METRICS_PATH = Path("D:/NCKH/SCAR/training/metrics/surface_distance.py")
assert SCAR_METRICS_PATH.is_file(), f"SCAR reference file not found at {SCAR_METRICS_PATH}"
_spec = importlib.util.spec_from_file_location("scar_surface_distance_ref", str(SCAR_METRICS_PATH))
scar_sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scar_sd)


# ==============================================================================
# 3D Shape Synthesis Fixtures & Helpers
# ==============================================================================

def make_torus_mask(shape: Tuple[int, int, int] = (48, 48, 24), R: float = 12.0, r: float = 4.0,
                    center: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """Generates a 3D torus mask: non-convex shape with an internal topological hole."""
    if center is None:
        center = (shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0)
    cx, cy, cz = center
    grid_x, grid_y, grid_z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    d_xy = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
    dist_torus = np.sqrt((d_xy - R) ** 2 + (grid_z - cz) ** 2)
    return dist_torus <= r


def make_c_shape_mask(shape: Tuple[int, int, int] = (48, 48, 24), r_inner: float = 6.0,
                      r_outer: float = 14.0, z_range: Tuple[int, int] = (4, 18),
                      gap_theta: float = np.pi / 2.0) -> np.ndarray:
    """Generates a non-convex C-shaped / horseshoe solid cylinder."""
    cx, cy = shape[0] / 2.0, shape[1] / 2.0
    grid_x, grid_y, grid_z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    r = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
    theta = np.arctan2(grid_y - cy, grid_x - cx)  # [-pi, pi]
    radial_mask = (r >= r_inner) & (r <= r_outer)
    z_mask = (grid_z >= z_range[0]) & (grid_z < z_range[1])
    angular_mask = np.abs(theta) > (gap_theta / 2.0)
    return radial_mask & z_mask & angular_mask


def make_concentric_shells_mask(shape: Tuple[int, int, int] = (48, 48, 32),
                                r1: float = 13.0, r2: float = 17.0,
                                r3: float = 5.0, r4: float = 9.0) -> np.ndarray:
    """Generates two concentric nested hollow shells with a gap and central cavity."""
    cx, cy, cz = shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0
    grid_x, grid_y, grid_z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2 + (grid_z - cz) ** 2)
    outer_shell = (dist >= r1) & (dist <= r2)
    inner_shell = (dist >= r3) & (dist <= r4)
    return outer_shell | inner_shell


def make_disconnected_islands_mask(shape: Tuple[int, int, int] = (48, 48, 24)) -> np.ndarray:
    """Generates multiple disconnected components and isolated satellite voxels."""
    mask = np.zeros(shape, dtype=bool)
    # Island 1: sphere at (12, 12, 6)
    gx, gy, gz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask |= (np.sqrt((gx - 12) ** 2 + (gy - 12) ** 2 + (gz - 6) ** 2) <= 4.0)
    # Island 2: cuboid at (34..42, 30..38, 12..18)
    mask[34:42, 30:38, 12:18] = True
    # Island 3: isolated single voxel
    mask[24, 40, 20] = True
    # Island 4: 2-voxel filament
    mask[8, 36, 10] = True
    mask[8, 37, 10] = True
    return mask


# ==============================================================================
# Suite 1: Direct Numerical Equivalence Against SCAR Reference
# ==============================================================================

class TestAdversarialSCARNumericalParity:
    """Adversarially stress-tests numerical equivalence against SCAR surface_distance.py."""

    TOLERANCE_OVERLAP = 1e-6
    TOLERANCE_DISTANCE = 1e-5

    def _assert_parity_with_scar(self, pred: np.ndarray, target: np.ndarray,
                                 spacing: Optional[Sequence[float]] = None,
                                 compute_distance: bool = True,
                                 empty_mode: str = "excluded"):
        """Helper to assert exact numerical equivalence between evaluate_predictions and SCAR."""
        # Note: SCAR default empty_mode is 'undefined', which behaves like 'excluded'
        scar_empty = "defined" if empty_mode == "defined" else "undefined"
        res_scar = scar_sd.binary_metrics(pred, target, spacing=spacing,
                                          compute_distance=compute_distance,
                                          empty_mode=scar_empty)
        res_ep = ep_binary_metrics(pred, target, spacing=spacing,
                                   compute_distance=compute_distance,
                                   empty_mode=empty_mode)

        # 1. Overlap metrics
        if res_scar["dice"] is None:
            assert res_ep["dice"] is None
        else:
            assert abs(res_scar["dice"] - res_ep["dice"]) < self.TOLERANCE_OVERLAP, (
                f"Dice mismatch: SCAR {res_scar['dice']} vs EP {res_ep['dice']}"
            )

        if res_scar["iou"] is None:
            assert res_ep["iou"] is None
        else:
            assert abs(res_scar["iou"] - res_ep["iou"]) < self.TOLERANCE_OVERLAP, (
                f"IoU mismatch: SCAR {res_scar['iou']} vs EP {res_ep['iou']}"
            )

        # 2. Distance metrics
        if res_scar["hd95"] is None:
            assert res_ep["hd95"] is None
        elif math.isinf(res_scar["hd95"]):
            assert math.isinf(res_ep["hd95"])
        else:
            assert abs(res_scar["hd95"] - res_ep["hd95"]) < self.TOLERANCE_DISTANCE, (
                f"HD95 mismatch: SCAR {res_scar['hd95']} vs EP {res_ep['hd95']}"
            )

        if res_scar["asd"] is None:
            assert res_ep["asd"] is None
        elif math.isinf(res_scar["asd"]):
            assert math.isinf(res_ep["asd"])
        else:
            assert abs(res_scar["asd"] - res_ep["asd"]) < self.TOLERANCE_DISTANCE, (
                f"ASSD/ASD mismatch: SCAR {res_scar['asd']} vs EP {res_ep['asd']}"
            )

        # 3. Status parity
        assert res_scar["status"] == res_ep["status"]

    def test_non_convex_torus_shape_parity(self):
        """Stress-test non-convex torus with internal hole and 3D translation/dilation."""
        target = make_torus_mask((48, 48, 24), R=12.0, r=4.5)
        # Prediction shifted by 1.5 voxels and dilated by 0.5 voxels
        pred = make_torus_mask((48, 48, 24), R=12.5, r=4.8, center=(25.5, 24.5, 12.0))

        # Test unit spacing
        self._assert_parity_with_scar(pred, target, spacing=None, compute_distance=True)

        # Test anisotropic clinical cardiac MRI spacing [0.75, 0.75, 4.0]
        self._assert_parity_with_scar(pred, target, spacing=(0.75, 0.75, 4.0), compute_distance=True)

    def test_c_shape_horseshoe_arc_parity(self):
        """Stress-test C-shaped / horseshoe solid cylinder with high surface curvature."""
        target = make_c_shape_mask((48, 48, 24), r_inner=6.0, r_outer=14.0, gap_theta=np.pi / 2.0)
        pred = make_c_shape_mask((48, 48, 24), r_inner=5.0, r_outer=15.0, gap_theta=np.pi / 2.5)

        self._assert_parity_with_scar(pred, target, spacing=None, compute_distance=True)
        self._assert_parity_with_scar(pred, target, spacing=(1.2, 1.2, 3.5), compute_distance=True)

    def test_concentric_nested_hollow_shells_parity(self):
        """Stress-test concentric nested hollow shells with multiple internal surfaces."""
        target = make_concentric_shells_mask((48, 48, 32), r1=13.0, r2=17.0, r3=5.0, r4=9.0)
        pred = make_concentric_shells_mask((48, 48, 32), r1=14.0, r2=18.0, r3=4.0, r4=8.0)

        self._assert_parity_with_scar(pred, target, spacing=None, compute_distance=True)
        self._assert_parity_with_scar(pred, target, spacing=(0.9, 0.9, 2.5), compute_distance=True)

    def test_multi_component_disconnected_islands_parity(self):
        """Stress-test multi-component disconnected volumes and isolated single voxels."""
        target = make_disconnected_islands_mask((48, 48, 24))
        # Create perturbed prediction with displaced components and false-positive cluster
        pred = np.roll(target, shift=(1, -1, 1), axis=(0, 1, 2))
        pred[20:25, 20:25, 5:8] = True  # Extra false positive cluster

        self._assert_parity_with_scar(pred, target, spacing=None, compute_distance=True)
        self._assert_parity_with_scar(pred, target, spacing=(1.0, 1.0, 5.0), compute_distance=True)

    def test_thin_structures_and_boundary_contacts_parity(self):
        """Stress-test 1-voxel planar sheets and masks touching volume edges."""
        shape = (32, 32, 16)
        pred = np.zeros(shape, dtype=bool)
        target = np.zeros(shape, dtype=bool)

        # 1-voxel thick planar sheet touching boundary z=0
        pred[5:25, 5:25, 0] = True
        target[8:28, 8:28, 0] = True

        # Corner touching blocks
        pred[0:4, 0:4, 0:2] = True
        target[0:3, 0:3, 0:2] = True

        self._assert_parity_with_scar(pred, target, spacing=None, compute_distance=True)
        self._assert_parity_with_scar(pred, target, spacing=(0.8, 0.8, 4.0), compute_distance=True)

    def test_randomized_monte_carlo_geometries_parity(self):
        """Stress-test 10 randomized 3D morphologies with Gaussian random fields."""
        rng = np.random.RandomState(42)
        shape = (32, 32, 16)

        for seed_idx in range(10):
            raw_field1 = rng.randn(*shape)
            raw_field2 = rng.randn(*shape)
            # Threshold to produce irregular blobs
            target = raw_field1 > 1.2
            pred = raw_field2 > 1.2

            # Guarantee non-empty
            target[10:15, 10:15, 5:8] = True
            pred[12:17, 12:17, 6:9] = True

            self._assert_parity_with_scar(pred, target, spacing=None, compute_distance=True)
            self._assert_parity_with_scar(pred, target, spacing=None, compute_distance=False)


# ==============================================================================
# Suite 2: Exhaustive Empty Mask Matrix Tests
# ==============================================================================

class TestAdversarialEmptyMaskMatrix:
    """Adversarially tests all permutations of empty masks under excluded vs defined modes."""

    def test_both_empty_excluded_mode(self):
        """Both empty under 'excluded' mode must yield None distances and None overlap."""
        pred = np.zeros((32, 32, 10), dtype=bool)
        target = np.zeros((32, 32, 10), dtype=bool)

        res = ep_binary_metrics(pred, target, empty_mode="excluded")
        assert res["status"] == "both_empty"
        assert res["dice"] is None
        assert res["iou"] is None
        assert res["hd95"] is None
        assert res["asd"] is None
        assert res["precision"] is None
        assert res["recall"] is None

    def test_both_empty_defined_mode(self):
        """Both empty under 'defined' mode must yield Dice=1.0, IoU=1.0, HD95=0.0, ASSD=0.0."""
        pred = np.zeros((32, 32, 10), dtype=bool)
        target = np.zeros((32, 32, 10), dtype=bool)

        res = ep_binary_metrics(pred, target, empty_mode="defined")
        assert res["status"] == "both_empty"
        assert res["dice"] == 1.0
        assert res["iou"] == 1.0
        assert res["hd95"] == 0.0
        assert res["asd"] == 0.0
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0

    def test_prediction_empty_target_full_excluded_mode(self):
        """Prediction empty, target full under 'excluded' mode -> Dice=0.0, distances None."""
        pred = np.zeros((32, 32, 10), dtype=bool)
        target = np.zeros((32, 32, 10), dtype=bool)
        target[10:20, 10:20, 2:8] = True

        res = ep_binary_metrics(pred, target, empty_mode="excluded")
        assert res["status"] == "prediction_empty"
        assert res["dice"] == 0.0
        assert res["iou"] == 0.0
        assert res["hd95"] is None
        assert res["asd"] is None
        assert res["precision"] is None
        assert res["recall"] == 0.0

    def test_prediction_empty_target_full_defined_mode(self):
        """Prediction empty, target full under 'defined' mode -> Dice=0.0, distances inf."""
        pred = np.zeros((32, 32, 10), dtype=bool)
        target = np.zeros((32, 32, 10), dtype=bool)
        target[10:20, 10:20, 2:8] = True

        res = ep_binary_metrics(pred, target, empty_mode="defined")
        assert res["status"] == "one_empty"
        assert res["dice"] == 0.0
        assert res["iou"] == 0.0
        assert math.isinf(res["hd95"])
        assert math.isinf(res["asd"])
        assert res["precision"] == 1.0
        assert res["recall"] == 0.0

    def test_target_empty_prediction_full_excluded_mode(self):
        """Target empty, prediction full under 'excluded' mode -> Dice=0.0, distances None."""
        pred = np.zeros((32, 32, 10), dtype=bool)
        target = np.zeros((32, 32, 10), dtype=bool)
        pred[10:20, 10:20, 2:8] = True

        res = ep_binary_metrics(pred, target, empty_mode="excluded")
        assert res["status"] == "target_empty"
        assert res["dice"] == 0.0
        assert res["iou"] == 0.0
        assert res["hd95"] is None
        assert res["asd"] is None
        assert res["precision"] == 0.0
        assert res["recall"] is None

    def test_target_empty_prediction_full_defined_mode(self):
        """Target empty, prediction full under 'defined' mode -> Dice=0.0, distances inf."""
        pred = np.zeros((32, 32, 10), dtype=bool)
        target = np.zeros((32, 32, 10), dtype=bool)
        pred[10:20, 10:20, 2:8] = True

        res = ep_binary_metrics(pred, target, empty_mode="defined")
        assert res["status"] == "one_empty"
        assert res["dice"] == 0.0
        assert res["iou"] == 0.0
        assert math.isinf(res["hd95"])
        assert math.isinf(res["asd"])
        assert res["precision"] == 0.0
        assert res["recall"] == 1.0

    def test_single_voxel_exact_adjacency_geometries(self):
        """Stress-test 1-voxel masks under coincident, face, edge, and corner adjacency."""
        shape = (16, 16, 16)

        # 1. Coincident: exact 1-voxel overlap
        t = np.zeros(shape, dtype=bool)
        p = np.zeros(shape, dtype=bool)
        t[8, 8, 8] = True
        p[8, 8, 8] = True
        res_coincident = ep_binary_metrics(p, t)
        assert res_coincident["dice"] == 1.0
        assert res_coincident["iou"] == 1.0
        assert res_coincident["hd95"] == 0.0
        assert res_coincident["asd"] == 0.0

        # 2. Face adjacent: offset (0, 0, 1), distance = 1.0 voxel
        p_face = np.zeros(shape, dtype=bool)
        p_face[8, 8, 9] = True
        res_face = ep_binary_metrics(p_face, t)
        assert res_face["dice"] == 0.0
        assert res_face["iou"] == 0.0
        assert abs(res_face["hd95"] - 1.0) < 1e-6
        assert abs(res_face["asd"] - 1.0) < 1e-6

        # 3. Edge adjacent: offset (0, 1, 1), distance = sqrt(2) ≈ 1.41421 voxels
        p_edge = np.zeros(shape, dtype=bool)
        p_edge[8, 9, 9] = True
        res_edge = ep_binary_metrics(p_edge, t)
        assert res_edge["dice"] == 0.0
        assert abs(res_edge["hd95"] - np.sqrt(2.0)) < 1e-5
        assert abs(res_edge["asd"] - np.sqrt(2.0)) < 1e-5

        # 4. Corner adjacent: offset (1, 1, 1), distance = sqrt(3) ≈ 1.73205 voxels
        p_corner = np.zeros(shape, dtype=bool)
        p_corner[9, 9, 9] = True
        res_corner = ep_binary_metrics(p_corner, t)
        assert res_corner["dice"] == 0.0
        assert abs(res_corner["hd95"] - np.sqrt(3.0)) < 1e-5
        assert abs(res_corner["asd"] - np.sqrt(3.0)) < 1e-5

        # 5. Face adjacent with physical spacing (0.5, 1.0, 4.0): offset in z has distance 4.0 mm
        res_aniso = ep_binary_metrics(p_face, t, spacing=(0.5, 1.0, 4.0))
        assert abs(res_aniso["hd95"] - 4.0) < 1e-5
        assert abs(res_aniso["asd"] - 4.0) < 1e-5


# ==============================================================================
# Suite 3: Target Region Disjointness & Hierarchical Completeness
# ==============================================================================

class TestAdversarialTargetRegionDisjointness:
    """Adversarially verifies disjointness, composition, and completeness of benchmark regions."""

    def test_canonical_region_definitions(self):
        """Verify the exact canonical label sets of the 5 benchmark target regions."""
        regions = BENCHMARK_PROTOCOL["regions"]
        assert regions["normal_myocardium"] == [1]
        assert regions["edema"] == [2]
        assert regions["scar"] == [3]
        assert regions["edema_inclusive"] == [2, 3]
        assert regions["myocardial_ring"] == [1, 2, 3]

    def test_pairwise_disjointness_of_elementary_regions(self):
        """Verify that elementary regions (normal, edema, scar) are strictly disjoint."""
        vol = np.random.RandomState(42).randint(0, 4, size=(32, 32, 16), dtype=np.uint8)

        mask_norm = np.isin(vol, BENCHMARK_PROTOCOL["regions"]["normal_myocardium"])
        mask_edema = np.isin(vol, BENCHMARK_PROTOCOL["regions"]["edema"])
        mask_scar = np.isin(vol, BENCHMARK_PROTOCOL["regions"]["scar"])

        # Pairwise intersections must be completely empty
        assert not np.logical_and(mask_norm, mask_edema).any()
        assert not np.logical_and(mask_norm, mask_scar).any()
        assert not np.logical_and(mask_edema, mask_scar).any()

    def test_hierarchical_composition_invariants(self):
        """Verify hierarchical union identities: edema_inclusive and myocardial_ring."""
        vol = np.random.RandomState(42).randint(0, 4, size=(32, 32, 16), dtype=np.uint8)

        mask_norm = np.isin(vol, BENCHMARK_PROTOCOL["regions"]["normal_myocardium"])
        mask_edema = np.isin(vol, BENCHMARK_PROTOCOL["regions"]["edema"])
        mask_scar = np.isin(vol, BENCHMARK_PROTOCOL["regions"]["scar"])
        mask_lge_abn = np.isin(vol, BENCHMARK_PROTOCOL["regions"]["edema_inclusive"])
        mask_myo = np.isin(vol, BENCHMARK_PROTOCOL["regions"]["myocardial_ring"])
        mask_bg = (vol == 0)

        # 1. edema | scar == edema_inclusive
        assert np.array_equal(mask_edema | mask_scar, mask_lge_abn)

        # 2. normal | edema_inclusive == myocardial_ring
        assert np.array_equal(mask_norm | mask_lge_abn, mask_myo)

        # 3. normal | edema | scar == myocardial_ring
        assert np.array_equal(mask_norm | mask_edema | mask_scar, mask_myo)

        # 4. Total volume partition completeness: bg + myo == whole volume
        assert np.array_equal(mask_bg | mask_myo, np.ones_like(vol, dtype=bool))
        assert not np.logical_and(mask_bg, mask_myo).any()

    def test_voxel_count_conservation_in_evaluate_single_volume(self):
        """Verify that voxel counts reported in detailed rows preserve exact partition arithmetic."""
        rng = np.random.RandomState(99)
        pred = rng.randint(0, 4, size=(24, 24, 12), dtype=np.uint8)
        raw_gt = rng.randint(0, 4, size=(24, 24, 12), dtype=np.uint8)

        detailed_rows, case_summary = evaluate_single_volume(pred, raw_gt, "case_test_counts")
        row_map = {r["region"]: r for r in detailed_rows}

        # Check prediction voxel conservation
        n_p_norm = row_map["normal_myocardium"]["prediction_voxels"]
        n_p_edema = row_map["edema"]["prediction_voxels"]
        n_p_scar = row_map["scar"]["prediction_voxels"]
        n_p_lge = row_map["edema_inclusive"]["prediction_voxels"]
        n_p_myo = row_map["myocardial_ring"]["prediction_voxels"]

        assert n_p_edema + n_p_scar == n_p_lge
        assert n_p_norm + n_p_edema + n_p_scar == n_p_myo

        # Check target voxel conservation
        n_t_norm = row_map["normal_myocardium"]["target_voxels"]
        n_t_edema = row_map["edema"]["target_voxels"]
        n_t_scar = row_map["scar"]["target_voxels"]
        n_t_lge = row_map["edema_inclusive"]["target_voxels"]
        n_t_myo = row_map["myocardial_ring"]["target_voxels"]

        assert n_t_edema + n_t_scar == n_t_lge
        assert n_t_norm + n_t_edema + n_t_scar == n_t_myo

    def test_region_aliases_consistency(self):
        """Verify that short aliases (MYO, EDEMA, SCAR, LGE-abn, NORM) match canonical fields exactly."""
        rng = np.random.RandomState(123)
        pred = rng.randint(0, 4, size=(20, 20, 10), dtype=np.uint8)
        raw_gt = rng.randint(0, 4, size=(20, 20, 10), dtype=np.uint8)

        _, summary = evaluate_single_volume(pred, raw_gt, "case_alias")

        assert summary["myo_dice"] == summary["myocardial_ring_dice"]
        assert summary["myo_iou"] == summary["myocardial_ring_iou"]
        assert summary["edema_dice"] == summary["edema_dice"]
        assert summary["scar_dice"] == summary["scar_dice"]
        assert summary["lge_abn_dice"] == summary["edema_inclusive_dice"]
        assert summary["norm_dice"] == summary["normal_myocardium_dice"]


# ==============================================================================
# Suite 4: Label Reconciliation Integrity Tests
# ==============================================================================

class TestAdversarialLabelReconciliation:
    """Adversarially stress-tests CANONICAL_LUT = [0, 1, 3, 2] against inversion and swap bugs."""

    def test_canonical_lut_specification(self):
        """Verify CANONICAL_LUT values, dtype, and elementary mappings."""
        assert isinstance(CANONICAL_LUT, np.ndarray)
        assert CANONICAL_LUT.dtype == np.uint8
        assert len(CANONICAL_LUT) == 4
        assert np.array_equal(CANONICAL_LUT, np.array([0, 1, 3, 2], dtype=np.uint8))

        # Check individual mappings
        assert CANONICAL_LUT[0] == 0  # Background
        assert CANONICAL_LUT[1] == 1  # Normal myocardium
        assert CANONICAL_LUT[2] == 3  # Raw Scar -> Canonical Scar
        assert CANONICAL_LUT[3] == 2  # Raw Edema -> Canonical Edema

    def test_canonical_lut_involution_property(self):
        """Verify CANONICAL_LUT is an involution (transposition of 2 and 3)."""
        labels = np.array([0, 1, 2, 3], dtype=np.uint8)
        # Applying LUT twice returns the original raw labels
        assert np.array_equal(CANONICAL_LUT[CANONICAL_LUT[labels]], labels)

    def test_label_reconciliation_identity_alignment(self):
        """Verify that a perfect model prediction achieves 1.0 Dice across all 5 regions."""
        # Ground truth has known anatomy in raw encoding
        raw_gt = np.zeros((30, 30, 10), dtype=np.uint8)
        raw_gt[5:15, 5:15, 2:8] = 1   # Raw Normal
        raw_gt[15:20, 15:20, 2:8] = 2 # Raw Scar
        raw_gt[20:25, 20:25, 2:8] = 3 # Raw Edema

        # Prediction output from model in canonical encoding:
        # Normal is 1, Edema is 2, Scar is 3
        pred = np.zeros((30, 30, 10), dtype=np.uint8)
        pred[5:15, 5:15, 2:8] = 1   # Canonical Normal
        pred[15:20, 15:20, 2:8] = 3 # Canonical Scar (matches raw 2)
        pred[20:25, 20:25, 2:8] = 2 # Canonical Edema (matches raw 3)

        _, case_summary = evaluate_single_volume(pred, raw_gt, "case_reconciliation_perfect")

        assert case_summary["normal_myocardium_dice"] == 1.0
        assert case_summary["edema_dice"] == 1.0
        assert case_summary["scar_dice"] == 1.0
        assert case_summary["edema_inclusive_dice"] == 1.0
        assert case_summary["myocardial_ring_dice"] == 1.0
        assert case_summary["avg_pathology_dice"] == 1.0
        assert case_summary["avg_pathology_inclusive_dice"] == 1.0

    def test_label_inversion_attack_sensitivity(self):
        """Adversarial Attack: If predictions predict raw labels instead of canonical,
        Scar Dice and Edema Dice must collapse to 0.0, while union regions remain 1.0."""
        raw_gt = np.zeros((30, 30, 10), dtype=np.uint8)
        raw_gt[10:15, 10:15, 2:8] = 2 # Raw Scar (5*5*6 = 150 voxels)
        raw_gt[20:25, 20:25, 2:8] = 3 # Raw Edema (5*5*6 = 150 voxels)

        # Inverted prediction: predicted Scar as 2 and Edema as 3 (unmapped bug)
        pred_inverted = np.zeros((30, 30, 10), dtype=np.uint8)
        pred_inverted[10:15, 10:15, 2:8] = 2 # Should be 3 for Scar!
        pred_inverted[20:25, 20:25, 2:8] = 3 # Should be 2 for Edema!

        _, case_summary = evaluate_single_volume(pred_inverted, raw_gt, "case_inversion_attack")

        # Pathology-specific Dice must collapse to 0.0
        assert case_summary["scar_dice"] == 0.0, "Inversion attack failed to detect Scar swap"
        assert case_summary["edema_dice"] == 0.0, "Inversion attack failed to detect Edema swap"
        assert case_summary["avg_pathology_dice"] == 0.0

        # But union regions must remain 1.0 because {2, 3} == {3, 2}
        assert case_summary["edema_inclusive_dice"] == 1.0
        assert case_summary["myocardial_ring_dice"] == 1.0


# ==============================================================================
# Suite 5: CSV & JSON Summary Statistics Mathematical Integrity
# ==============================================================================

class TestAdversarialCSVSummaryMathematicalIntegrity:
    """Adversarially tests mathematical integrity of per_case.csv and metrics.json on 76 cases."""

    @pytest.fixture
    def synthetic_76_case_cohort(self, tmp_path: Path) -> Tuple[Path, Path, Path, Path]:
        """Constructs a full 76-patient test cohort with diverse pathologies and empty masks."""
        pred_dir = tmp_path / "predictions"
        data_root = tmp_path / "data"
        split_dir = tmp_path / "splits"
        output_dir = tmp_path / "evaluation"

        pred_dir.mkdir(parents=True, exist_ok=True)
        split_dir.mkdir(parents=True, exist_ok=True)
        h5_dir = data_root / "bSSFP" / "test_vol_h5"
        h5_dir.mkdir(parents=True, exist_ok=True)

        case_ids = [f"case{i:04d}" for i in range(1, 77)]
        test_vol_txt = split_dir / "test_vol.txt"
        test_vol_txt.write_text("\n".join(case_ids) + "\n", encoding="utf-8")

        rng = np.random.RandomState(42)
        vol_shape = (24, 24, 8)

        for idx, cid in enumerate(case_ids):
            pred = np.zeros(vol_shape, dtype=np.uint8)
            gt_raw = np.zeros(vol_shape, dtype=np.uint8)

            # Normal myocardium present in all cases
            pred[4:18, 4:18, 1:7] = 1
            gt_raw[4:18, 4:18, 1:7] = 1

            # Sub-cohort distribution to stress empty mask logic:
            # Cases 0..24: Both Scar and Edema present
            if idx < 25:
                pred[6:10, 6:10, 2:5] = 3   # Scar
                gt_raw[6:10, 6:10, 2:5] = 2 # Raw Scar
                pred[12:16, 12:16, 2:5] = 2  # Edema
                gt_raw[12:16, 12:16, 2:5] = 3# Raw Edema
            # Cases 25..44: Scar-only (Edema both_empty)
            elif idx < 45:
                pred[6:10, 6:10, 2:5] = 3
                gt_raw[6:10, 6:10, 2:5] = 2
                # Edema is empty in both
            # Cases 45..59: Edema-only (Scar both_empty)
            elif idx < 60:
                pred[12:16, 12:16, 2:5] = 2
                gt_raw[12:16, 12:16, 2:5] = 3
                # Scar is empty in both
            # Cases 60..69: Purely normal (both Scar and Edema both_empty)
            elif idx < 70:
                pass
            # Cases 70..72: Prediction false positives (Target empty for Scar)
            elif idx < 73:
                pred[6:10, 6:10, 2:5] = 3 # FP Scar
            # Cases 73..75: Prediction false negatives (Target present, Pred empty for Scar)
            else:
                gt_raw[6:10, 6:10, 2:5] = 2 # FN Scar

            # Save prediction .npz
            np.savez_compressed(pred_dir / f"{cid}_pred.npz", prediction=pred)

            # Save ground truth .npy.h5
            with h5py.File(h5_dir / f"{cid}.npy.h5", "w") as f:
                f.create_dataset("label", data=gt_raw.astype(np.float32))

        return pred_dir, data_root, test_vol_txt, output_dir

    def test_full_76_case_evaluation_and_summary_statistics(
        self, synthetic_76_case_cohort: Tuple[Path, Path, Path, Path]
    ):
        """End-to-end evaluation on 76 cases: verifies per_case.csv, detailed.csv, and metrics.json."""
        pred_dir, data_root, split_file, output_dir = synthetic_76_case_cohort

        summary = evaluate_dataset(
            predictions_dir=pred_dir,
            data_root=data_root,
            split_file=split_file,
            output_dir=output_dir,
            compute_distance=True,
            empty_mode="excluded",
        )

        per_case_csv = output_dir / "per_case.csv"
        detailed_csv = output_dir / "per_case_detailed.csv"
        metrics_json = output_dir / "metrics.json"

        assert per_case_csv.is_file(), "per_case.csv was not created"
        assert detailed_csv.is_file(), "per_case_detailed.csv was not created"
        assert metrics_json.is_file(), "metrics.json was not created"

        # ----------------------------------------------------------------------
        # 1. Verify per_case_detailed.csv: exactly 76 * 5 = 380 rows
        # ----------------------------------------------------------------------
        with open(detailed_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            detailed_rows = list(reader)

        assert len(detailed_rows) == 76 * 5 == 380, f"Expected 380 detailed rows, got {len(detailed_rows)}"

        # Cross-check directly against SCAR's reference summarize_rows function
        typed_detailed_rows = []
        for r in detailed_rows:
            tr = dict(r)
            for k in ("dice", "iou", "precision", "recall", "hd95_voxel", "asd_voxel"):
                tr[k] = float(r[k]) if r[k] != "" else None
            typed_detailed_rows.append(tr)

        scar_summary = scar_sd.summarize_rows(typed_detailed_rows)

        # ----------------------------------------------------------------------
        # 2. Verify metrics.json mathematical parity against SCAR summarize_rows
        # ----------------------------------------------------------------------
        with open(metrics_json, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        for reg in ("myocardial_ring", "edema", "scar", "edema_inclusive", "normal_myocardium"):
            scar_st = scar_summary[reg]
            ep_st = json_data[reg]

            # Cases count
            assert ep_st["cases"] == scar_st["cases"] == 76

            # Defined/undefined counts
            assert ep_st["dice_defined_cases"] == scar_st["dice_defined_cases"]
            assert ep_st["dice_undefined_cases"] == scar_st["dice_undefined_cases"]
            assert ep_st["dice_defined_cases"] + ep_st["dice_undefined_cases"] == 76

            # Mean Dice
            if scar_st["mean_dice"] is None:
                assert ep_st["mean_dice"] is None
            else:
                assert abs(scar_st["mean_dice"] - ep_st["mean_dice"]) < 1e-10

            # Mean HD95
            if scar_st["mean_hd95_voxel"] is None:
                assert ep_st["mean_hd95_voxel"] is None
            else:
                assert abs(scar_st["mean_hd95_voxel"] - ep_st["mean_hd95_voxel"]) < 1e-10

        # Composite pathology dice
        assert abs(scar_summary["avg_pathology_inclusive_dice"] - json_data["avg_pathology_inclusive_dice"]) < 1e-10
        assert abs(scar_summary["avg_pathology_dice"] - json_data["avg_pathology_dice"]) < 1e-10

        # ----------------------------------------------------------------------
        # 3. Verify per_case.csv exact NumPy parity for Mean, Std, Median rows
        # ----------------------------------------------------------------------
        with open(per_case_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        # Expected: 76 patient rows + 3 summary rows (Mean, Std, Median) = 79 rows
        assert len(csv_rows) == 79, f"Expected 79 rows in per_case.csv, got {len(csv_rows)}"

        patient_rows = csv_rows[:76]
        summary_rows = {r["case_id"]: r for r in csv_rows[76:]}

        assert "Mean" in summary_rows
        assert "Std" in summary_rows
        assert "Median" in summary_rows

        row_mean = summary_rows["Mean"]
        row_std = summary_rows["Std"]
        row_median = summary_rows["Median"]

        # Check every metric column across patient rows
        fieldnames = [k for k in csv_rows[0].keys() if k != "case_id"]
        for field in fieldnames:
            if field.endswith("_status"):
                assert row_mean[field] == ""
                assert row_std[field] == ""
                assert row_median[field] == ""
                continue

            # Collect numeric values from patient rows
            patient_vals = [float(r[field]) for r in patient_rows if r[field] != ""]

            if len(patient_vals) == 0:
                assert row_mean[field] == ""
                assert row_std[field] == ""
                assert row_median[field] == ""
            else:
                expected_mean = float(np.mean(patient_vals))
                expected_std = float(np.std(patient_vals))
                expected_median = float(np.median(patient_vals))

                actual_mean = float(row_mean[field])
                actual_std = float(row_std[field])
                actual_median = float(row_median[field])

                assert abs(actual_mean - expected_mean) < 1e-5, (
                    f"Mean mismatch for {field}: actual {actual_mean} vs expected {expected_mean}"
                )
                assert abs(actual_std - expected_std) < 1e-5, (
                    f"Std mismatch for {field}: actual {actual_std} vs expected {expected_std}"
                )
                assert abs(actual_median - expected_median) < 1e-5, (
                    f"Median mismatch for {field}: actual {actual_median} vs expected {expected_median}"
                )


# ==============================================================================
# Suite 6: Input Validation, Degradation & Spacing Stress Tests
# ==============================================================================

class TestAdversarialInputValidationAndDegradation:
    """Stress-tests error handling, edge cases, and graceful degradation."""

    def test_invalid_spacing_validation(self):
        """Verify strict validation of physical spacing arguments."""
        p = np.ones((10, 10, 10), dtype=bool)
        t = np.ones((10, 10, 10), dtype=bool)

        # 1. Negative spacing
        with pytest.raises(ValueError, match="Spacing must contain"):
            ep_binary_metrics(p, t, spacing=(1.0, -1.0, 1.0))

        # 2. Zero spacing
        with pytest.raises(ValueError, match="Spacing must contain"):
            ep_binary_metrics(p, t, spacing=(1.0, 0.0, 1.0))

        # 3. Non-finite spacing (NaN / inf)
        with pytest.raises(ValueError, match="Spacing must contain"):
            ep_binary_metrics(p, t, spacing=(1.0, float("nan"), 1.0))

        # 4. Dimension mismatch (2 elements for 3D array)
        with pytest.raises(ValueError, match="Spacing must contain"):
            ep_binary_metrics(p, t, spacing=(1.0, 1.0))

    def test_shape_mismatch_validation(self):
        """Verify shape mismatch raises ValueError."""
        p = np.zeros((16, 16, 8), dtype=bool)
        t = np.zeros((16, 16, 10), dtype=bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            ep_binary_metrics(p, t)

    def test_out_of_bounds_target_label_raises_index_error(self):
        """Verify that ground truth with invalid labels (>3) raises IndexError via CANONICAL_LUT."""
        p = np.zeros((10, 10, 10), dtype=np.uint8)
        raw_gt = np.zeros((10, 10, 10), dtype=np.uint8)
        raw_gt[2, 2, 2] = 4  # Invalid label outside [0..3]

        with pytest.raises(IndexError):
            evaluate_single_volume(p, raw_gt, "case_invalid_label")

    def test_all_cases_undefined_pathology_graceful_degradation(self, tmp_path: Path):
        """Verify graceful degradation when an entire cohort completely lacks a pathology.
        If NO cases have edema (all both_empty), summary must report mean=None, undefined=76,
        avg_pathology_dice=None, and CSV summary rows must output empty string without crashing."""
        pred_dir = tmp_path / "pred_no_edema"
        data_root = tmp_path / "data_no_edema"
        split_dir = tmp_path / "splits_no_edema"
        output_dir = tmp_path / "eval_no_edema"

        pred_dir.mkdir(parents=True, exist_ok=True)
        split_dir.mkdir(parents=True, exist_ok=True)
        h5_dir = data_root / "bSSFP" / "test_vol_h5"
        h5_dir.mkdir(parents=True, exist_ok=True)

        case_ids = [f"case{i:04d}" for i in range(1, 11)]  # 10 cases
        test_vol_txt = split_dir / "test_vol.txt"
        test_vol_txt.write_text("\n".join(case_ids) + "\n", encoding="utf-8")

        for cid in case_ids:
            pred = np.zeros((16, 16, 8), dtype=np.uint8)
            gt_raw = np.zeros((16, 16, 8), dtype=np.uint8)
            # Only Normal and Scar; Edema is completely empty
            pred[2:8, 2:8, 2:6] = 1
            gt_raw[2:8, 2:8, 2:6] = 1
            pred[3:5, 3:5, 3:5] = 3
            gt_raw[3:5, 3:5, 3:5] = 2

            np.savez_compressed(pred_dir / f"{cid}_pred.npz", prediction=pred)
            with h5py.File(h5_dir / f"{cid}.npy.h5", "w") as f:
                f.create_dataset("label", data=gt_raw.astype(np.float32))

        summary = evaluate_dataset(
            predictions_dir=pred_dir,
            data_root=data_root,
            split_file=test_vol_txt,
            output_dir=output_dir,
            compute_distance=True,
            empty_mode="excluded",
        )

        # Edema should have 0 defined cases and 10 undefined cases
        assert summary["edema"]["dice_defined_cases"] == 0
        assert summary["edema"]["dice_undefined_cases"] == 10
        assert summary["edema"]["mean_dice"] is None

        # Composite pathology dice requires both; should be None
        assert summary["avg_pathology_dice"] is None

        # But scar is defined
        assert summary["scar"]["mean_dice"] == 1.0

        # Verify per_case.csv summary rows output empty string for undefined edema metrics
        with open(output_dir / "per_case.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {r["case_id"]: r for r in reader}

        assert rows["Mean"]["edema_dice"] == ""
        assert rows["Std"]["edema_dice"] == ""
        assert rows["Median"]["edema_dice"] == ""
        assert float(rows["Mean"]["scar_dice"]) == 1.0

    def test_physical_mm_spacing_output_schema(self, tmp_path: Path):
        """Verify that supplying --spacing records distance_unit='mm' and hd95_mm fields."""
        pred_dir = tmp_path / "pred_mm"
        data_root = tmp_path / "data_mm"
        split_dir = tmp_path / "splits_mm"
        output_dir = tmp_path / "eval_mm"

        pred_dir.mkdir(parents=True, exist_ok=True)
        split_dir.mkdir(parents=True, exist_ok=True)
        h5_dir = data_root / "bSSFP" / "test_vol_h5"
        h5_dir.mkdir(parents=True, exist_ok=True)

        cid = "case0001"
        test_vol_txt = split_dir / "test_vol.txt"
        test_vol_txt.write_text(f"{cid}\n", encoding="utf-8")

        pred = np.zeros((16, 16, 8), dtype=np.uint8)
        gt_raw = np.zeros((16, 16, 8), dtype=np.uint8)
        pred[2:8, 2:8, 2:6] = 1
        gt_raw[2:8, 2:8, 2:6] = 1

        np.savez_compressed(pred_dir / f"{cid}_pred.npz", prediction=pred)
        with h5py.File(h5_dir / f"{cid}.npy.h5", "w") as f:
            f.create_dataset("label", data=gt_raw.astype(np.float32))

        summary = evaluate_dataset(
            predictions_dir=pred_dir,
            data_root=data_root,
            split_file=test_vol_txt,
            output_dir=output_dir,
            spacing=(0.8, 0.8, 4.0),
            compute_distance=True,
            empty_mode="excluded",
        )

        assert summary["distance_unit"] == "mm"
        assert summary["metadata"]["distance_unit"] == "mm"

        # Check per_case_detailed.csv contains hd95_mm
        with open(output_dir / "per_case_detailed.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first_row = next(reader)

        assert first_row["hd95_unit"] == "mm"
        assert "hd95_mm" in first_row
        assert "asd_mm" in first_row

