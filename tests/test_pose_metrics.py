import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

np = pytest.importorskip("numpy")

from metrics.pose_metrics import (
    f1_depth,
    mpjpe2d,
    mpjpe3d,
    mean_average_precision,
    pca,
)


def test_f1_depth():
    pred = np.array([[0.05, 0.12], [0.15, 0.20]])
    gt = np.array([[0.04, 0.20], [0.10, 0.07]])
    result = f1_depth(pred, gt, threshold=0.1)
    assert result == pytest.approx(2 / 3)


def test_mpjpe2d():
    pred = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    gt = np.array([[[1.0, 1.0], [2.0, 4.0]]])
    assert mpjpe2d(pred, gt) == pytest.approx(1.0)


def test_mpjpe3d():
    pred = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    gt = np.array([[[0.0, 0.0, 1.0], [1.0, 2.0, 1.0]]])
    assert mpjpe3d(pred, gt) == pytest.approx(1.0)


def test_mean_average_precision():
    pred = np.array([[[0.0, 0.0]], [[1.0, 0.0]]])
    gt = np.array([[[0.0, 0.0]], [[2.0, 0.0]]])
    scores = np.array([[0.9], [0.8]])
    result = mean_average_precision(pred, gt, scores, thresholds=[1.0])
    assert result == pytest.approx(1.0)


def test_pca():
    data = np.array([[1.0, 0.0], [0.0, 1.0]])
    eigvals, eigvecs = pca(data)
    assert eigvals.shape == (2,)
    assert eigvecs.shape == (2, 2)
    assert eigvals[0] == pytest.approx(1.0)
    assert eigvals[1] == pytest.approx(0.0)
    expected_abs = 1 / math.sqrt(2)
    assert np.allclose(np.abs(eigvecs), expected_abs)
