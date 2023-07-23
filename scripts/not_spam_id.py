import re
import yaml
import pandas as pd
import fire


def job():
    """
    Reads the config file and extracts the necessary information to generate a CSV file with the not spam IDs.

    Returns:
        None
    """
    with open("./config.yml", "r", encoding="utf8") as config_file:
        config = yaml.safe_load(config_file)

        path_not_spam_id = config["path_not_spam_id"]

        not_spam_path = config["not_spam_path"]

    not_spam_csv = pd.read_csv(f"data/{not_spam_path}", sep=";")

    mask_not_spam = not_spam_csv["label"] == 0
    list_not_spam = not_spam_csv[mask_not_spam]["from_id"].unique().tolist()
    list_not_spam_id = [
        "".join(re.findall("[0-9]", str(from_id))) for from_id in list_not_spam
    ]

    pd.DataFrame({"not_spam_id": list_not_spam_id}).to_csv(
        f"{path_not_spam_id}", sep=";"
    )


if __name__ == "__main__":
    fire.Fire(job)
