import yaml
import fire
from src.data.data import Data


def job():
    data_path = "data"

    with open("./config.yml", "r", encoding="utf8") as config_file:
        config = yaml.safe_load(config_file)

        json_not_spam_path = config["json_not_spam_path"]
        json_spam_path = config["json_spam_path"]

        not_spam_path = config["not_spam_path"]
        spam_path = config["spam_path"]

        cleaned_not_spam_path = config["cleaned_not_spam_path"]
        cleaned_spam_path = config["cleaned_spam_path"]

        train_path = config["train_path"]
        test_path = config["test_path"]

    Data.json_to_csv(
        f"{data_path}/{json_not_spam_path}", f"{data_path}/{not_spam_path}", 0
    )
    Data.json_to_csv(f"{data_path}/{json_spam_path}", f"{data_path}/{spam_path}", 1)

    Data.clean_dataframe(
        f"{data_path}/{not_spam_path}", f"{data_path}/{cleaned_not_spam_path}"
    )
    Data.clean_dataframe(f"{data_path}/{spam_path}", f"{data_path}/{cleaned_spam_path}")

    list_clean_dataframe = [
        f"{data_path}/{cleaned_not_spam_path}",
        f"{data_path}/{cleaned_spam_path}",
    ]
    Data.train_test_split(
        list_clean_dataframe, f"{data_path}/{train_path}", f"{data_path}/{test_path}"
    )


if __name__ == "__main__":
    fire.Fire(job)
