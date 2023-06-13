import pandas as pd
import yaml
import fire
from rules_base_model import Model  # import your Model class from its file


def job() -> None:
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
        file_path = config["df_cleaned_spam_and_not_spam"]
        save_path = config["labels_and_scores"]
    df_cleaned_spam_and_not_spam = pd.read_csv(file_path, sep=";")
    labels_and_scores = pd.read_csv(save_path, sep=";")

    model = Model()
    pred_scores = model.predict(df_cleaned_spam_and_not_spam["text"])
    labels_and_scores["pred_scores"] = pred_scores

    labels_and_scores = pd.DataFrame(
        {
            "text": df_cleaned_spam_and_not_spam["text"],
            "pred_scores": pred_scores,
            "label": df_cleaned_spam_and_not_spam["label"].values,
        }
    )
    labels_and_scores.to_csv(save_path, sep=";", index=False)


if __name__ == "__main__":
    fire.Fire(job)
