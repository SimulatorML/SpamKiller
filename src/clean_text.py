import re
import pandas as pd
import numpy as np


def clean_dataframe(file_path, save_path):
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


# # Using the function to clear the dataframe and save it along the specified path
clean_dataframe(
    "data/text_spam_dataset/not_spam.csv", "data/text_spam_dataset/cleaned_not_spam.csv"
)
clean_dataframe(
    "data/text_spam_dataset/spam.csv", "data/text_spam_dataset/cleaned_spam.csv"
)
