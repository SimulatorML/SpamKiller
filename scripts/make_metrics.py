import yaml
import pandas as pd
import numpy as np
import fire
from src.metrics.metrics import (
    RecallAtPrecision,
    RecallAtSpecificity,
    ConfusionMatrix,
    BootstrapCurve,
)


def job(
    min_value: float = 0.99, conf: float = 0.95, n_bootstraps: int = 10_000
) -> None:
    """
    Runs a job to evaluate the performance of a binary classification model based on the provided parameters.
    The job reads data from a config file, sets a threshold value, and computes evaluation metrics
    using the Metrics class and the provided parameters.

    :param min_precision: Minimum required precision for the model. Defaults to 0.95.
    :type min_precision: float
    :param min_specificity: Minimum required specificity for the model. Defaults to 0.95.
    :type min_specificity: float
    :param conf: Confidence level for the confidence interval. Defaults to 0.95.
    :type conf: float
    :param n_bootstraps: Number of bootstrap samples for the confidence interval. Defaults to 10_000.
    :type n_bootstraps: int

    :return: None
    :rtype: None
    """

    with open("./config.yml", "r", encoding="utf8") as config_file:
        config = yaml.safe_load(config_file)
        labels_and_scores = config["labels_and_scores"]

    labels_and_scores = pd.read_csv(
        labels_and_scores,
        sep=";",
    )

    true_labels = labels_and_scores["label"].to_numpy()
    pred_scores = labels_and_scores["pred_scores"].to_numpy()

    recall_at_precision = RecallAtPrecision(
        min_value=min_value, conf=conf, n_bootstraps=n_bootstraps
    )
    recall_at_specificity = RecallAtSpecificity(
        min_value=min_value, conf=conf, n_bootstraps=n_bootstraps
    )

    conf_matrix = ConfusionMatrix()
    precision_recall_curve = BootstrapCurve(
        metric="precision_recall", conf=conf, n_bootstraps=n_bootstraps
    )
    specificity_recall_curve = BootstrapCurve(
        metric="specificity_recall", conf=conf, n_bootstraps=n_bootstraps
    )

    threshold, max_recall_at_precision = recall_at_precision.max_recall(
        true_labels, pred_scores
    )
    (
        lcb_recall_at_precision,
        ucb_recall_at_precision,
    ) = recall_at_precision.bootstrap_recall(true_labels, pred_scores)
    _, max_recall_at_specificity = recall_at_specificity.max_recall(
        true_labels, pred_scores
    )
    (
        lcb_recall_at_specificity,
        ucb_recall_at_specificity,
    ) = recall_at_specificity.bootstrap_recall(true_labels, pred_scores)

    pred_labels = np.where(pred_scores < threshold, 0, 1)

    precision_recall_curve(true_labels, pred_scores)
    specificity_recall_curve(true_labels, pred_scores)
    conf_matrix(true_labels, pred_labels)

    print("recall@precision %.2f%% : %.2f" % (min_value * 100, max_recall_at_precision))

    print(
        "lower: %s and upper: %s bounds in a confidence interval (%s) "
        "of a recall@precision %.2f%% : %.2f"
        % (
            lcb_recall_at_precision,
            ucb_recall_at_precision,
            conf,
            min_value * 100,
            max_recall_at_precision,
        )
    )

    print(
        "recall@specificity %.2f%% : %.2f"
        % (min_value * 100, max_recall_at_specificity)
    )

    print(
        "lower: %s and upper: %s bounds in a confidence interval (%s) "
        "of a recall@specificity %.2f%% : %.2f"
        % (
            lcb_recall_at_specificity,
            ucb_recall_at_specificity,
            conf,
            min_value * 100,
            max_recall_at_specificity,
        )
    )


if __name__ == "__main__":
    fire.Fire(job)
