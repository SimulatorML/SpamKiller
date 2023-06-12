import pandas as pd
import yaml
import fire
from src.models.rule_based_model import RuleBasedClassifier


def job(Model=RuleBasedClassifier) -> None:
    """
    Runs a given model on a cleaned dataset of spam and non-spam messages, and saves the predicted scores
    and corresponding labels to a CSV file.

    :param Model: A model class that has a `predict` method.
    :type Model: class

    :return: None
    :rtype: NoneType
    """
    with open("./config.yml", "r", encoding="utf8") as config_file:
        config = yaml.safe_load(config_file)
        train_path = config["train"]
        test_path = config["test"]
        save_path = config["labels_and_scores"]

    train = pd.read_csv(train_path, sep=";")
    test = pd.read_csv(test_path, sep=";")

    model = Model()
    model.fit(train[["text", "photo"]], train["label"])
    pred_scores = model.predict(test[["text", "photo"]])

    labels_and_scores = pd.DataFrame(
        {"pred_scores": pred_scores, "label": test["label"].values}
    )
    labels_and_scores.to_csv(save_path, sep=";", index=False)


if __name__ == "__main__":
    fire.Fire(job)
