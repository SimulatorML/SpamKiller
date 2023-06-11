import fire
from src.data.data import Data


def job():
    """
    Converts json data to csv, cleans the csv, and preprocesses the data for
    classification. Does not take any parameters and does not return anything.
    """
    # convert json data to csv
    Data.json_to_csv(
        "data/text_spam_dataset", "result_not_spam.json", "not_spam.csv", 0
    )
    Data.json_to_csv("data/text_spam_dataset", "result_spam.json", "spam.csv", 1)

    # clean data
    Data.clean_dataframe(
        "data/text_spam_dataset/not_spam.csv",
        "data/text_spam_dataset/cleaned_not_spam.csv",
    )
    Data.clean_dataframe(
        "data/text_spam_dataset/spam.csv", "data/text_spam_dataset/cleaned_spam.csv"
    )

    # final preprocessing
    Data.dataset()


if __name__ == "__main__":
    fire.Fire(job)
