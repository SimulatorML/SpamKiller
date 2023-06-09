from typing import Any
from typing import Tuple
from typing import Callable

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from tqdm.notebook import tqdm


def binary_search_last_greater(arr, value) -> int:
    """Calculates the last value which >= value

    Args:
        arr (np.ndarray): Array to search
        value (float): Target value

    Returns:
        int: Value index
    """
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:  # до тех пор пока индексы не сошлись
        mid = left + (right - left) // 2  # находим средний индекс
        if arr[mid] >= value:  # если значение больше или равно таргету
            result = mid  # индекс становить равен среднему
            left = mid + 1  # левый индекс становиться средний плюс один
        else:
            right = mid - 1  # правый индекс становиться средний минус один
    return result


def recall_at_precision(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_precision: float = 0.95,
) -> float:
    """Compute recall at precision

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_precision (float, optional): Min precision for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    mask = precision >= min_precision
    metric = recall[mask].max()
    return metric

def recall_at_specificity(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_specificity: float = 0.95,
) -> float:
    """Compute recall at specificity

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_specificity (float, optional): Min specificity for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """
    fpr, tpr, _ = roc_curve(true_labels, pred_scores, drop_intermediate=False)
    specificity = 1 - fpr
    index = binary_search_last_greater(
        specificity, min_specificity
    )  # используем бинарный поиск для поиска последнего элемента
    if index == -1:  # если индекс не нашелся, возвращаем ноль
        return 0.0
    metric = tpr[index]
    return metric

def bootstrap_metric(
    metric: Callable,
    label: np.ndarray,
    probability: np.ndarray,
    min_value: float = 0.95,
    conf: float = 0.95,
    n_bootstraps: int = 10000,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Returns confidence bounds of the metric

    Args:
        metric (function): function for bootstrap
        label (np.ndarray): True labels
        probability (np.ndarray): Model scores
        min_value (float): Min value for metric,
        conf (float): Confidence interval,
        n_bootstraps (int): Sampling amount,
        verbose (bool): Tqdm verbose

    Returns:
        Tuple[float, float]: lcb and ucb metric
    """
    list_score = []
    range_label = np.arange(len(label))
    bootstraps_index = np.random.choice(
        range_label, size=len(label) * n_bootstraps, replace=True
    )
    mask = label == 1
    index_class_1 = range_label[mask]
    for index in tqdm(range(n_bootstraps), disable=not verbose):
        try:
            bottom_line = index * len(label)
            upper_line = len(label) * (index + 1)
            bootstrap = bootstraps_index[bottom_line:upper_line]
            bootstraps_class_1 = np.random.choice(index_class_1, size=1)
            bootstrap = np.r_[
                bootstraps_class_1, bootstrap
            ]  # добавляем один индекс класса 1, из-за сильного дисбаланса
            score = metric(label[bootstrap], probability[bootstrap], min_value)
            list_score.append(score)
        except ValueError:
            continue
    bound = (1 - conf) / 2
    lcb = np.nanquantile(list_score, bound)
    ucb = np.nanquantile(list_score, 1 - bound)
    return (lcb, ucb)


def curves(true_labels: np.ndarray, pred_scores: np.ndarray) -> Tuple[np.ndarray]:
    """Return ROC and FPR curves

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores

    Returns:
        Tuple[np.ndarray]: ROC and FPR curves
    """
    def fig2numpy(fig: Any) -> np.ndarray:
        fig.canvas.draw()
        img = fig.canvas.buffer_rgba()
        img = np.asarray(img)
        return img

    pr_curve = PrecisionRecallDisplay.from_predictions(
        true_labels, pred_scores
    )  # строим PrecisionRecall фигуру
    pr_curve = fig2numpy(
        pr_curve.figure_
    )  # переводим PrecisionRecall фигуру в numpy.array

    roc_curve = RocCurveDisplay.from_predictions(
        true_labels, pred_scores
    )  # строим RocCurve фигуру
    roc_curve = fig2numpy(roc_curve.figure_)  # переводим RocCurve фигуру в numpy.array

    return pr_curve, roc_curve
