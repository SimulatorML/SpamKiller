import json
import csv
import os
import re
import pandas as pd
import numpy as np
import yaml


class Data:
    """
    This class contains methods for loading, cleaning, and processing data used in the SpamKiller project.
    """

    @staticmethod
    def json_to_csv(json_path: str, csv_path: str, label: int) -> None:
        """
        Function for extracting text from a JSON file and writing it to a CSV file

        Parameters
        ----------
        json_path : str
            Full path to the JSON file
        csv_path : str
            Full path to the CSV file

        Returns
        -------
        None

        """

        # Opening the JSON file
        with open(json_path, encoding="utf-8", newline="") as f:
            data = json.load(f)

        # Extracting the text from the JSON file
        messages_to_write = []
        for message in data["messages"]:
            text = message.get("text")
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
            messages_to_write.append([text, label])

        # Writing the text to the CSV file
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(["text", "label"])  # Writing the header
            writer.writerows(messages_to_write)

        print(
            f"Ready! The text from the JSON file was overwritten into a file {csv_path}"
        )

    @staticmethod
    def clean_dataframe(file_path: str, save_path: str) -> pd.DataFrame:
        """
        Function for cleaning a dataframe from a CSV file

        Parameters
        ----------
        file_path : str
            Path to the CSV file
        save_path : str
            Path to save the cleaned dataframe

        Returns
        -------
        dataframe : pandas.core.frame.DataFrame
            Cleaned dataframe

        """

        # Loading a dataframe from a CSV file using ';' as a separator
        dataframe = pd.read_csv(file_path, sep=";", na_values=["nan"])
        # Deleting rows with empty values
        dataframe = dataframe.dropna()
        # Cleaning of symbols and emoticons
        # dataframe['text'] = dataframe['text'].apply(
        #     lambda x: re.sub(r'[^\w\s]', '', str(x)))
        # Replacing numbers with a special token
        # dataframe['text'] = dataframe['text'].str.replace(
        #     '\d+', '<NUMBER>', regex=True)
        # Cast to lowercase
        # dataframe['text'] = dataframe['text'].str.lower()
        # Removing extra spaces at the beginning, end and in the text itself
        dataframe["text"] = (
            dataframe["text"].str.replace("\s+", " ", regex=True).str.strip()
        )
        # Removing duplicates
        dataframe = dataframe.drop_duplicates(subset="text")
        # Remove rows with empty strings
        dataframe = dataframe[dataframe["text"].str.len() > 0]
        # Saving the cleared dataframe along the specified path
        dataframe.to_csv(save_path, index=False, sep=";")
        return dataframe

    @staticmethod
    def dataset() -> None:
        """
        This staticmethod loads dataset from CSV files specified in the config.yml file.
        It concatenates two dataframes and saves them as a CSV file specified in the config.yml.
        :return: None
        """
        with open("./config.yml", "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            cleaned_spam_path = config["path_cleaned_spam"]
            clened_not_spam_path = config["path_not_spam"]
            df_cleaned_spam_and_not_spam_path = config["df_cleaned_spam_and_not_spam"]

        # Load data cleaned_spam from CSV
        cleaned_spam = pd.read_csv(cleaned_spam_path, sep=";")

        # Load data clened_not_spam from CSV
        clened_not_spam = pd.read_csv(clened_not_spam_path, sep=";")

        # Concatenating two dataframes
        df_cleaned_spam_and_not_spam = pd.concat(
            [cleaned_spam, clened_not_spam], ignore_index=True
        )
        # Drop NaN values
        df_cleaned_spam_and_not_spam = df_cleaned_spam_and_not_spam.dropna()

        # Save dataframe to CSV
        df_cleaned_spam_and_not_spam.to_csv(
            df_cleaned_spam_and_not_spam_path, sep=";", index=False
        )


if __name__ == "__main__":
    # Initialize your Data class
    data = Data()

    with open("./config.yml", "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
        json_spam = config["json_spam"]
        json_not_spam = config["json_not_spam"]
        path_spam = config["path_spam"]
        path_cleaned_spam = config["path_cleaned_spam"]
        path_not_spam = config["path_not_spam"]
        path_cleaned_not_spam = config["path_cleaned_not_spam"]

    # Call json_to_csv
    data.json_to_csv(json_spam, path_spam, 1)
    data.json_to_csv(json_not_spam, path_not_spam, 0)

    # # Call clean_dataframe
    cleaned_spam_dataframe = data.clean_dataframe(path_spam, path_cleaned_spam)
    cleaned_not_spam_dataframe = data.clean_dataframe(
        path_not_spam, path_cleaned_not_spam
    )
    # Call dataset
    data.dataset()
