from typing import Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


@dataclass
class Metrics(ABC):
    """Metrics"""

    min_value: float = 0.99
    conf: float = 0.99
    n_bootstraps: int = 10000
    verbose: bool = False

    @abstractmethod
    def max_recall(self, true_labels: np.ndarray, pred_scores: np.ndarray) -> float:
        """
        This is an abstract method that calculates the maximum recall score for a binary classification problem.
        The function takes in two parameters:
        `true_labels` is a numpy array that contains the true labels for the binary classification problem.
        `pred_scores` is a numpy array that contains the predicted scores for the binary classification problem.
        The function returns a float value that represents the maximum recall score for the binary classification problem.
        """
        pass

    def bootstrap_recall(
        self, true_labels: np.ndarray, pred_scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute the lower and upper confidence bounds of the recall metric using bootstrap resampling.

        Args:
            true_labels (np.ndarray): Array of ground truth labels.
            pred_scores (np.ndarray): Array of predicted scores.

        Returns:
            Tuple[float, float]: A tuple containing the lower and upper confidence bounds of the recall metric.
        """
        list_score = []
        range_label = np.arange(len(true_labels))
        bootstraps_index = np.random.choice(
            range_label, size=len(true_labels) * self.n_bootstraps, replace=True
        )

        for index in tqdm(range(self.n_bootstraps), disable=not self.verbose):
            try:
                bottom_line = index * len(true_labels)
                upper_line = len(true_labels) * (index + 1)
                bootstrap = bootstraps_index[bottom_line:upper_line]
                _, recall = self.max_recall(
                    true_labels[bootstrap], pred_scores[bootstrap]
                )
                list_score.append(recall)

            except ValueError:
                continue

        bound = (1 - self.conf) / 2
        lcb = np.quantile(list_score, bound)
        ucb = np.quantile(list_score, 1 - bound)

        return (lcb, ucb)


@dataclass
class RecallAtPrecision(Metrics):
    """Precision at recall"""

    def max_recall(
        self, true_labels: np.ndarray, pred_scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculates the maximum recall and corresponding threshold probability for a binary classification problem.

        Args:
            true_labels (np.ndarray): A numpy array of true labels.
            pred_scores (np.ndarray): A numpy array of predicted scores.

        Returns:
            A tuple of two floats representing the threshold probability and max recall.
        """
        try:
            argsort_array = np.argsort(pred_scores)[::-1]
            thresholds = pred_scores[argsort_array]
            labels = true_labels[argsort_array]

            distinct_value_indices = np.where(np.diff(thresholds))[0]
            threshold_idxs = np.r_[distinct_value_indices, labels.size - 1]

            thresholds = thresholds[threshold_idxs]
            tps = np.cumsum(labels)[threshold_idxs]
            fps = np.cumsum(1 - labels)[threshold_idxs]
            precision = tps / (tps + fps)
            recall = tps / tps[-1]

            mask = precision >= self.min_value
            max_recall = np.max(recall[mask])
            mask2 = recall == max_recall
            threshold_proba = sorted(thresholds[mask2], reverse=True)[0]

        except ValueError:
            max_recall = 0.0
            threshold_proba = 0.0

        return threshold_proba, max_recall


@dataclass
class RecallAtSpecificity(Metrics):
    """Recall at specificity"""

    def max_recall(
        self, true_labels: np.ndarray, pred_scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculates the maximum recall given true labels and predicted scores.
        :param true_labels: A numpy array of true labels.
        :param pred_scores: A numpy array of predicted scores.
        :return: A tuple containing the threshold probability and maximum recall.
        """
        try:
            argsort_array = np.argsort(pred_scores)[::-1]
            thresholds = pred_scores[argsort_array]
            labels = true_labels[argsort_array]

            distinct_value_indices = np.where(np.diff(thresholds))[0]
            threshold_idxs = np.r_[distinct_value_indices, labels.size - 1]

            thresholds = thresholds[threshold_idxs]
            tps = np.cumsum(labels)[threshold_idxs]
            fp = np.cumsum(1 - labels)[threshold_idxs]
            specificity = 1 - fp / fp[-1]
            recall = tps / tps[-1]

            index = self._binary_search_last_greater(specificity, self.min_value)

            if index == -1:
                return 0.0, 0.0

            max_recall = recall[index]
            threshold_proba = thresholds[index]

        except:
            max_recall = 0.0
            threshold_proba = 0.0

        return threshold_proba, max_recall

    def _binary_search_last_greater(self, arr: np.ndarray, value: float) -> int:
        """Calculates the last value which is >= value

        Args:
            arr (np.ndarray): Array to search
            value (float): Target value

        Returns:
            int: Value index
        """
        left, right = 0, len(arr) - 1
        result = -1

        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] >= value:
                result = mid
                left = mid + 1
            else:
                right = mid - 1

        return result


@dataclass
class Plot(ABC):
    """Base class for plotting"""

    @abstractmethod
    def __call__(self, true_labels: np.ndarray, pred_scores: np.ndarray) -> None:
        """
        This is an abstract method that takes in two numpy arrays of true labels and predicted scores,
        respectively. It does not return anything. It is meant to be overridden by a subclass to define
        a custom behavior for evaluating the predicted scores against the true labels.

        Parameters:
            true_labels (np.ndarray): A 1-dimensional numpy array containing true labels.
            pred_scores (np.ndarray): A 2-dimensional numpy array containing predicted scores.

        Returns:
            None
        """
        pass


@dataclass
class ConfusionMatrix(Plot):
    """Plot a confusion matrix"""

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list = "default",
        ymap: dict = "default",
    ) -> None:
        """
        Generate a confusion matrix heatmap plot using seaborn and matplotlib.

        Parameters:
        -----------
        y_true : np.ndarray
            An array of true labels.
        y_pred : np.ndarray
            An array of predicted labels.
        labels : list, optional
            The label names to be used in the plot. Default is ["0", "1"].
        ymap : dict, optional
            A dictionary mapping the label values to their corresponding names.
            Default is {1: "Spam", 0: "Not spam"}.

        Returns:
        --------
        None
        """
        if labels == "default":
            labels = [0, 1]
        if ymap == "default":
            ymap = {1: "Spam", 0: "Not spam"}
        if ymap is not None:
            y_pred = [ymap[yi] for yi in y_pred]
            y_true = [ymap[yi] for yi in y_true]
            labels = [ymap[yi] for yi in labels]

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape

        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
                elif c == 0:
                    annot[i, j] = ""
                else:
                    annot[i, j] = "%.1f%%\n%d" % (p, c)

        annot[0, 0] += "\nTN"
        annot[0, 1] += "\nFP"
        annot[1, 0] += "\nFN"
        annot[1, 1] += "\nTP"

        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm.index.name = "Actual"
        cm.columns.name = "Predicted"
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=annot, fmt="", ax=ax)
        plt.show()


@dataclass
class BootstrapCurve(Plot):
    """Plot a bootstrap curve"""

    metric: str = "precision_recall"
    conf: float = 0.95
    n_bootstraps: int = 10_000

    def __call__(self, true_labels: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Compute the precision-recall or specificity-recall curve with a confidence interval using bootstrapping.

        Parameters:
        true_labels: np.ndarray - array of true labels
        y_pred: np.ndarray - array of predicted probabilities
        n_bootstraps: int - number of bootstrap iterations
        metric: str - "precision_recall" or "specificity_recall"
        conf: float - confidence level

        Returns:
        None - plots the curve and its confidence interval
        """
        range_label = np.arange(len(true_labels))
        bootstraps_index = np.random.choice(
            range_label,
            size=true_labels.shape[0] * self.n_bootstraps,
            replace=True,
        )

        argsort_array = np.argsort(y_pred)[::-1]
        true_labels = true_labels[argsort_array]

        if self.metric == "precision_recall":
            tps = np.cumsum(true_labels)
            fps = np.cumsum(1 - true_labels)
            y = tps / (tps + fps)
            x = tps / tps[-1]

        if self.metric == "specificity_recall":
            tps = np.cumsum(true_labels)
            fps = np.cumsum(1 - true_labels)
            y = 1 - fps / fps[-1]
            x = tps / tps[-1]

        bootstraps = []
        for iter in range(self.n_bootstraps):
            bootstraps_index_down = iter * true_labels.shape[0]
            bootstraps_index_up = iter * true_labels.shape[0] + true_labels.shape[0]
            bootstrap = bootstraps_index[bootstraps_index_down:bootstraps_index_up]

            y_pred_for_bootsrap = y_pred[bootstrap]
            argsort_for_bootsrap = np.argsort(y_pred_for_bootsrap)[::-1]
            y_true_for_bootsrap = true_labels[bootstrap][argsort_for_bootsrap]

            if self.metric == "precision_recall":
                tps = np.cumsum(y_true_for_bootsrap)
                fps = np.cumsum(1 - y_true_for_bootsrap)
                y_bootstrap = tps / (tps + fps)
                bootstraps.append(y_bootstrap)

            if self.metric == "specificity_recall":
                fps = np.cumsum(1 - y_true_for_bootsrap)
                y_bootstrap = 1 - fps / fps[-1]
                bootstraps.append(y_bootstrap)

        matrix_bootstraps = np.vstack(bootstraps)
        bound = (1 - self.conf) / 2
        y_lcb = np.quantile(matrix_bootstraps, bound, axis=0)
        y_ucb = np.quantile(matrix_bootstraps, 1 - bound, axis=0)

        x = np.concatenate(([0], x))
        y = np.concatenate(([1], y))
        y_lcb = np.concatenate(([1], y_lcb))
        y_ucb = np.concatenate(([1], y_ucb))

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label=f"{self.metric}", color="navy")
        plt.fill_between(
            x, y_lcb, y_ucb, color="skyblue", alpha=0.5, label=f"{self.conf*100}% CI"
        )
        plt.title(f"Plot of {self.metric} with {self.conf*100}% Confidence Interval")
        plt.xlabel("Recall")
        plt.ylabel("Precision" if self.metric == "precision_recall" else "Specificity")
        plt.legend()
        plt.show()


def calculate_recall(y_true, y_pred):
    """
    Calculate the recall, also known as sensitivity or true positive rate.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels in binary classification.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels in binary classification.

    Returns
    -------
    recall : float
        The recall metric as a float.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def calculate_specificity(y_true, y_pred):
    """
    Calculate the specificity, also known as true negative rate.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels in binary classification.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels in binary classification.

    Returns
    -------
    specificity : float
        The specificity metric as a float.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def calculate_precision(y_true, y_pred):
    """
    Calculate the precision, also known as the positive predictive value.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels in binary classification.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels in binary classification.

    Returns
    -------
    precision : float
        The precision metric as a float.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fp) if (tp + fp) > 0 else 0