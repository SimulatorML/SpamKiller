import json
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


class Data:
    @staticmethod
    def json_to_csv(json_dir: str, csv_dir: str, label: int) -> None:
        """
        Converts a JSON file to a CSV file.

        Parameters:
            json_dir (str): The directory path of the JSON file.
            csv_dir (str): The directory path to save the CSV file.
            label (int): The label to assign to each message in the CSV file.

        Returns:
            None
        """
        with open(json_dir, encoding="utf-8", newline="") as f:
            data = json.load(f)

        messages_to_write = []
        for message in data["messages"]:
            text = message.get("text")
            from_id = message.get("from_id")
            photo = "photo" in message
            reply_to_message_id = "reply_to_message_id" in message
            if isinstance(text, list):
                texts = []
                for entity in text:
                    if isinstance(entity, str):
                        texts.append(entity)
                    elif isinstance(entity, dict):
                        if "text" in entity and isinstance(entity["text"], str):
                            texts.append(entity["text"])
                text = " ".join(texts)
            elif not isinstance(text, str):
                text = ""
            messages_to_write.append([text, photo, from_id, reply_to_message_id, label])

        with open(csv_dir, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(
                ["text", "photo", "from_id", "reply_to_message_id", "label"]
            )
            writer.writerows(messages_to_write)

        print(
            f"Ready! The text from the JSON file was overwritten into a file {csv_dir}"
        )

    @staticmethod
    def clean_dataframe(file_dir: str, save_dir: str) -> None:
        """
        Clean the given dataframe by removing null values, duplicate text, and empty text entries.

        Parameters:
            file_dir (str): The directory of the CSV file to read.
            save_dir (str): The directory to save the cleaned dataframe as a CSV file.

        Returns:
            None
        """
        dataframe = pd.read_csv(file_dir, sep=";")

        dataframe = dataframe.dropna()
        dataframe["text"] = (
            dataframe["text"].str.replace("\s+", " ", regex=True).str.strip()
        )
        dataframe = dataframe.drop_duplicates(subset="text")
        dataframe = dataframe[dataframe["text"].str.len() > 0]

        dataframe.to_csv(save_dir, index=False, sep=";")

    @staticmethod
    def train_test_split(files_dir: list, train_dir: str, test_dir: str) -> None:
        """
        Split the given list of file directories into train and test datasets.

        Parameters:
            - files_dir (list): A list of file directories.
            - train_dir (str): The directory to save the train dataset.
            - test_dir (str): The directory to save the test dataset.

        Returns:
            None
        """
        train = []
        test = []

        for path in files_dir:
            df = pd.read_csv(path, sep=";")
            train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

            train_df.reset_index(drop=True, inplace=True)
            test_df.reset_index(drop=True, inplace=True)

            train.append(train_df)
            test.append(test_df)

        train = pd.concat(train, ignore_index=True, axis=0)
        test = pd.concat(test, ignore_index=True, axis=0)

        train = train.dropna().drop_duplicates(subset="text")
        test = test.dropna().drop_duplicates(subset="text")

        train.to_csv(train_dir, sep=";", index=False)
        test.to_csv(test_dir, sep=";", index=False)
