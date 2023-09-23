import pandas as pd
import yaml
import fire
from src.models.rules_base_model_validation import RuleBasedClassifier
from src.models.logistic_regression import SpamLogisticRegression


def job(Model=RuleBasedClassifier) -> None:
    """
    Runs a given model on a cleaned dataset of spam and non-spam messages, and saves the predicted scores
    and corresponding labels to a CSV file.

    :param Model: A model class that has a `predict` method.
    :type Model: class

    :return: None
    :rtype: NoneType
    """
    data_path = "data"

    with open("./config.yml", "r", encoding="utf8") as config_file:
        config = yaml.safe_load(config_file)
        train_path = config["train_path"]
        test_path = config["test_path"]
        save_path = config["labels_and_scores"]

    train = pd.read_csv(f"{data_path}/{train_path}", sep=";")
    test = pd.read_csv(f"{data_path}/{test_path}", sep=";")

    model = Model()
    if isinstance(model, SpamLogisticRegression):
        model.fit(train.drop(columns="label"), train["label"])
        pred_scores = model.predict(test.drop(columns="label"))
    else:
        model.fit(train.drop(columns="label"), train["label"])
        pred_scores, _ = model.predict(test.drop(columns="label"))

    labels_and_scores = pd.DataFrame(
        {"pred_scores": pred_scores, "label": test["label"].values}
    )
    labels_and_scores.to_csv(f"{data_path}/{save_path}", sep=";", index=False)


if __name__ == "__main__":
    fire.Fire(job)
