import yaml
import pandas as pd
import numpy as np
import fire
from src.metrics.metrics import Metrics


def job(
    min_precision: float = 0.95,
    min_specificity: float = 0.95,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
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

    # Read data
    with open("./config.yml", "r", encoding="utf8") as config_file:
        config = yaml.safe_load(config_file)
        labels_and_scores_path = config["labels_and_scores"]

    labels_and_scores = pd.read_csv(
        labels_and_scores_path,
        sep=";",
    )

    # prepare data
    true_labels = labels_and_scores["label"].to_numpy()
    pred_scores = labels_and_scores["pred_scores"].to_numpy()

    _, threshold = Metrics.recall_at_specificity(true_labels, pred_scores)
    pred_labels = np.where(pred_scores < threshold, 0, 1)

    Metrics.get_metrics(
        true_labels,
        pred_labels,
        pred_scores,
        min_precision,
        min_specificity,
        conf,
        n_bootstraps,
    )


if __name__ == "__main__":
    fire.Fire(job)
