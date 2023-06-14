from typing import Any, Tuple, Callable
import logging

from sklearn.metrics import (
    precision_recall_curve,
    PrecisionRecallDisplay,
    roc_curve,
    RocCurveDisplay,
    confusion_matrix,
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetricsCalculator")


class Metrics:
    """
    Class for calculating evaluation metrics for classification models.
    """

    @staticmethod
    def get_metrics(
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        min_precision: float = 0.95,
        min_specificity: float = 0.95,
        conf: float = 0.95,
        n_bootstraps: int = 10000,
        verbose: bool = False,
    ) -> None:
        """
        Calculate evaluation metrics and display the results.

        Args:
            true_labels (np.ndarray): True labels.
            pred_labels (np.ndarray): Predicted labels.
            pred_scores (np.ndarray): Predicted scores or probabilities.
            min_precision (float, optional): Minimum precision for recall calculation. Defaults to 0.95.
            min_specificity (float, optional): Minimum specificity for recall calculation. Defaults to 0.95.
            conf (float, optional): Confidence level for bootstrap confidence interval. Defaults to 0.95.
            n_bootstraps (int, optional): Number of bootstrap samples for confidence interval estimation. Defaults to 10000.
            verbose (bool, optional): Enable/disable progress bar. Defaults to False.

        Returns:
            None
        """
        rec_at_pre, _ = __class__.recall_at_precision(
            true_labels, pred_scores, min_precision
        )
        rec_at_spec, _ = __class__.recall_at_specificity(
            true_labels, pred_scores, min_specificity
        )
        lcb_rec_at_pre, ucb_rec_at_pre = __class__.bootstrap_metric(
            __class__.recall_at_precision,
            true_labels,
            pred_scores,
            min_precision,
            conf,
            n_bootstraps,
            verbose,
        )
        lcb_rec_at_spec, ucb_rec_at_spec = __class__.bootstrap_metric(
            __class__.recall_at_specificity,
            true_labels,
            pred_scores,
            min_specificity,
            conf,
            n_bootstraps,
            verbose,
        )

        logger.info("recall@precision %.2f%% : %.2f", min_precision * 100, rec_at_pre)

        logger.info(
            "lower: %s and upper: %s bounds in a confidence interval (%s) "
            "of a recall@precision %.2f%% : %.2f",
            lcb_rec_at_pre,
            ucb_rec_at_pre,
            conf,
            min_precision * 100,
            rec_at_pre,
        )

        logger.info(
            "recall@specificity %.2f%% : %.2f", min_specificity * 100, rec_at_spec
        )

        logger.info(
            "lower: %s and upper: %s bounds in a confidence interval (%s) "
            "of a recall@specificity %.2f%% : %.2f",
            lcb_rec_at_spec,
            ucb_rec_at_spec,
            conf,
            min_specificity * 100,
            rec_at_spec,
        )

        PrecisionRecallDisplay.from_predictions(true_labels, pred_scores)
        RocCurveDisplay.from_predictions(true_labels, pred_scores)
        plt.show()

        __class__.construction_confusion_matrix(true_labels, pred_labels)

    @staticmethod
    def recall_at_precision(
        true_labels: np.ndarray,
        pred_scores: np.ndarray,
        min_precision: float = 0.95,
    ) -> Tuple[float, None]:
        """Compute recall at precision

        Args:
            true_labels (np.ndarray): True labels
            pred_scores (np.ndarray): Target scores
            min_precision (float, optional): Minimum precision for recall. Defaults to 0.95.

        Returns:
            float: Metric value and None
        """
        precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
        mask = precision >= min_precision
        metric = recall[mask].max()
        treshold = None  # Not used, present for API consistency by convention.
        return metric, treshold

    @staticmethod
    def recall_at_specificity(
        true_labels: np.ndarray,
        pred_scores: np.ndarray,
        min_specificity: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute recall at specificity and treshold

        Args:
            true_labels (np.ndarray): True labels
            pred_scores (np.ndarray): Target scores
            min_specificity (float, optional): Minimum specificity for recall. Defaults to 0.95.

        Returns:
            float: Metric value
        """
        fpr, tpr, thresholds = roc_curve(
            true_labels, pred_scores, drop_intermediate=False
        )
        specificity = 1 - fpr
        index = __class__._binary_search_last_greater(
            specificity, min_specificity
        )  # Use binary search to find the last element
        if index == -1 or index == 0:  # If index not found, return 0
            return 0.0, 0.0
        metric = tpr[index]
        treshold = thresholds[index]
        return metric, treshold

    @staticmethod
    def bootstrap_metric(
        metric: Callable,
        label: np.ndarray,
        scores: np.ndarray,
        min_value: float = 0.95,
        conf: float = 0.95,
        n_bootstraps: int = 10000,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """Returns confidence bounds of the metric

        Args:
            metric (function): Function for bootstrap
            label (np.ndarray): True labels
            scores (np.ndarray): Model scores
            min_value (float): Minimum value for the metric
            conf (float): Confidence interval
            n_bootstraps (int): Sampling amount
            verbose (bool): Tqdm verbose

        Returns:
            Tuple[float, float]: Lower and upper confidence bounds of the metric
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
                ]  # Add one index of class 1 due to severe class imbalance
                probability, _ = metric(label[bootstrap], scores[bootstrap], min_value)
                list_score.append(probability)
            except ValueError:
                continue
        bound = (1 - conf) / 2
        lcb = np.nanquantile(list_score, bound)
        ucb = np.nanquantile(list_score, 1 - bound)
        return (lcb, ucb)

    @staticmethod
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
        )  # Generate Precision-Recall figure
        pr_curve = fig2numpy(
            pr_curve.figure_
        )  # Convert Precision-Recall figure to numpy.array

        rc_curve = RocCurveDisplay.from_predictions(
            true_labels, pred_scores
        )  # Generate ROC curve figure
        rc_curve = fig2numpy(
            rc_curve.figure_
        )  # Convert ROC curve figure to numpy.array

        return pr_curve, rc_curve

    @staticmethod
    def construction_confusion_matrix(
        true_labels: np.ndarray, predictions_labels: np.ndarray
    ) -> None:
        """
        Outputs a confusion matrix
        Args:
            true_labels (np.ndarray): True labels
            predictions_labels (np.ndarray): Target scores
        """
        conf_matrix = confusion_matrix(true_labels, predictions_labels)
        ax = sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g")

        # Создание подписей
        labels = np.array(
            [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
        )

        # Добавление подписей на график
        for i in range(2):
            for j in range(2):
                ax.text(
                    j + 0.5,
                    i + 0.8,
                    labels[i, j],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    @staticmethod
    def _binary_search_last_greater(arr: np.ndarray, value: float) -> int:
        """Calculates the last value which is >= value

        Args:
            arr (np.ndarray): Array to search
            value (float): Target value

        Returns:
            int: Value index
        """
        left, right = 0, len(arr) - 1
        result = -1
        while left <= right:  # Repeat until indices converge
            mid = left + (right - left) // 2  # Find the middle index
            if arr[mid] >= value:  # If the value is greater than or equal to the target
                result = mid  # Set the index to the middle
                left = mid + 1  # Update the left index to middle + 1
            else:
                right = mid - 1  # Update the right index to middle - 1
        return result
