import numpy as np
from typing import Iterable, Tuple


def f1_depth(pred_depth: np.ndarray, gt_depth: np.ndarray, threshold: float = 0.1) -> float:
    """Compute F1 score for depth classification.

    A joint is considered positive if its depth value is below the given
    threshold. The F1 score is computed between the predicted and ground truth
    classifications.

    Parameters
    ----------
    pred_depth: np.ndarray
        Predicted depths with shape (N, J).
    gt_depth: np.ndarray
        Ground truth depths with shape (N, J).
    threshold: float, optional
        Threshold separating near and far joints.

    Returns
    -------
    float
        F1 score over all joints.
    """
    pred_pos = pred_depth < threshold
    gt_pos = gt_depth < threshold

    tp = np.sum(pred_pos & gt_pos)
    fp = np.sum(pred_pos & ~gt_pos)
    fn = np.sum(~pred_pos & gt_pos)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def mpjpe2d(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per joint position error in 2D.

    Parameters
    ----------
    pred : np.ndarray
        Predicted 2D coordinates with shape (N, J, 2).
    gt : np.ndarray
        Ground truth 2D coordinates with shape (N, J, 2).

    Returns
    -------
    float
        The mean Euclidean distance over all joints.
    """
    diff = pred - gt
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return float(np.mean(dist))


def mpjpe3d(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per joint position error in 3D.

    Parameters
    ----------
    pred : np.ndarray
        Predicted 3D coordinates with shape (N, J, 3).
    gt : np.ndarray
        Ground truth 3D coordinates with shape (N, J, 3).

    Returns
    -------
    float
        The mean Euclidean distance over all joints.
    """
    diff = pred - gt
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return float(np.mean(dist))


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute the average precision (AP) given precision and recall arrays."""
    # Append sentinel values to compute area under curve
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Ensure precision is monotonically decreasing
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap


def mean_average_precision(
    pred: np.ndarray,
    gt: np.ndarray,
    scores: np.ndarray,
    thresholds: Iterable[float] = (0.05, 0.1, 0.2),
) -> float:
    """Compute mean Average Precision for keypoint predictions.

    Parameters
    ----------
    pred : np.ndarray
        Predicted keypoints with shape (N, J, D).
    gt : np.ndarray
        Ground truth keypoints with shape (N, J, D).
    scores : np.ndarray
        Confidence scores for predictions with shape (N, J).
    thresholds : Iterable[float], optional
        Distance thresholds for a match.

    Returns
    -------
    float
        Mean AP over all joints and thresholds.
    """
    n_samples, n_joints, _ = pred.shape
    aps = []

    for thr in thresholds:
        for j in range(n_joints):
            # Flatten predictions for joint j
            dists = np.linalg.norm(pred[:, j] - gt[:, j], axis=1)
            order = np.argsort(-scores[:, j])
            dists = dists[order]
            true_pos = dists <= thr

            tp = np.cumsum(true_pos)
            fp = np.cumsum(~true_pos)

            recall = tp / (n_samples + 1e-8)
            precision = tp / (tp + fp + 1e-8)

            ap = _average_precision(recall, precision)
            aps.append(ap)

    return float(np.mean(aps))


def pca(data: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Principal Component Analysis.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (N, D).
    n_components : int, optional
        Number of principal components to return. If ``None``, all components
        are returned.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Eigenvalues and eigenvectors sorted by descending eigenvalue.
    """
    if data.ndim != 2:
        raise ValueError("data must be 2D array")

    mean = np.mean(data, axis=0)
    centered = data - mean
    cov = np.cov(centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if n_components is not None:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    return eigvals, eigvecs

