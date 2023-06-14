import json
import csv
import os
import re
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


class Data:
    """
    This class contains methods for loading, cleaning, and processing data used in the SpamKiller project.
    """

    @staticmethod
    def json_to_csv(
        json_dir: str, json_filename: str, csv_filename: str, label: int
    ) -> None:
        """
        Function for extracting text from a JSON file and writing it to a CSV file

        Parameters
        ----------
        json_dir : str
            Relative path to the directory with the JSON file
        json_filename : str
            JSON file name
        csv_filename : str
            CSV file name

        Returns
        -------
        None

        """

        # Relative path to the directory with the JSON file
        json_path = os.path.join(json_dir, json_filename)

        # Opening the JSON file
        with open(json_path, encoding="utf-8", newline="") as f:
            data = json.load(f)

        # Relative path to the directory with the CSV file
        csv_path = os.path.join(json_dir, csv_filename)

        # Extracting the text from the JSON file
        messages_to_write = []
        for message in data["messages"]:
            text = message.get("text")
            # If the text is a list, concatenate all the 'text' from entities
            if isinstance(
                text, list
            ):  # If the text is a list, concatenate all the 'text' from entities
                texts = []
                for entity in text:  # Iterating over the list
                    if isinstance(
                        entity, str
                    ):  # If the element is a string, add it to the list
                        texts.append(
                            entity
                        )  # If the element is a string, add it to the list
                    elif isinstance(
                        entity, dict
                    ):  # If the element is a dictionary, check if it has a 'text' key
                        if "text" in entity and isinstance(
                            entity["text"], str
                        ):  # If the key exists and the value is a string, add it to the list
                            texts.append(
                                entity["text"]
                            )  # If the key exists and the value is a string, add it to the list
                text = " ".join(texts)  # Concatenating all the 'text' from entities
            elif not isinstance(
                text, str
            ):  # If not a string, convert it to an empty string
                text = ""
            messages_to_write.append(
                [text, "photo" in message, label]
            )  # Adding the text and label to the list

        # Writing the text to the CSV file
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(["text", "photo", "label"])  # Writing the header
            writer.writerows(messages_to_write)

        print(
            f"Ready! The text from the JSON file was overwritten into a file {csv_filename}"
        )

    @staticmethod
    def clean_dataframe(file_path: str, save_path: str) -> None:
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
        dataframe = pd.read_csv(file_path, sep=";")
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
            clened_not_spam_path = config["path_cleaned_not_spam"]
            train_path = config["path_train"]
            test_path = config["path_test"]

        # Load data cleaned_spam from CSV
        cleaned_spam = pd.read_csv(cleaned_spam_path, sep=";")

        # Load data clened_not_spam from CSV
        clened_not_spam = pd.read_csv(clened_not_spam_path, sep=";")

        train_spam, test_spam = train_test_split(
            cleaned_spam, test_size=0.2, shuffle=False
        )

        train_not_spam, test_not_spam = train_test_split(
            clened_not_spam, test_size=0.2, shuffle=False
        )
        train_spam.reset_index(drop=True, inplace=True)
        train_not_spam.reset_index(drop=True, inplace=True)
        test_spam.reset_index(drop=True, inplace=True)
        test_not_spam.reset_index(drop=True, inplace=True)

        # Concatenating two dataframes
        train = pd.concat([train_spam, train_not_spam], ignore_index=True, axis=0)

        test = pd.concat([test_spam, test_not_spam], ignore_index=True, axis=0)

        train = train.dropna().drop_duplicates(subset="text")  # надо здесь разобраться
        test = test.dropna().drop_duplicates(subset="text")

        # Saving the concatenated dataframes to CSV
        train.to_csv(train_path, sep=";", index=False)
        test.to_csv(test_path, sep=";", index=False)
