import yaml
import pandas as pd


with open("./config.yml", "r", encoding="utf8") as config_file:
    config = yaml.safe_load(config_file)

    path_not_spam_id = config["path_not_spam_id"]

    train_path = config["train_path"]

X = pd.read_csv(f"data/{train_path}", sep=";")

mask_not_spam = X["label"] == 0
z = X[mask_not_spam]["from_id"].unique()

pd.DataFrame({"from_id": z}).to_csv(f"{path_not_spam_id}", sep=";")
