import pandas as pd
import yaml
import fire
from tqdm import tqdm
from src.models.rules_base_model import (
    RuleBasedClassifier,
)  # import your Model class from its file


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
        test_path = config["path_test"]
        path_save_test = config["path_save_test"]
    test_path = pd.read_csv(test_path, sep=";")

    # Load the model
    model = Model()
    # Make predictions fof the test set
    pred_scores = model.predict(tqdm(test_path["text"]))
    test_path["pred_scores"] = pred_scores

    test_path = pd.DataFrame(
        {
            "text": test_path["text"],
            "pred_scores": pred_scores,
            "label": test_path["label"].values,
        }
    )

    test_path.to_csv(path_save_test, sep=";", index=False)


if __name__ == "__main__":
    fire.Fire(job)
