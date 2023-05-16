import json
import csv
import os


def get_csv(json_dir: str, json_filename: str, csv_filename: str, label: int):
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
    with open(json_path) as f:
        data = json.load(f)

    # Relative path to the directory with the CSV file
    csv_path = os.path.join(json_dir, csv_filename)

    # Extracting the text from the JSON file
    messages_to_write = []
    for message in data['messages']:
        text = message.get('text')
        # If the text is a list, concatenate all the 'text' from entities
        if isinstance(text, list): # If the text is a list, concatenate all the 'text' from entities
            texts = []
            for entity in text: # Iterating over the list
                if isinstance(entity, str): # If the element is a string, add it to the list
                    texts.append(entity) # If the element is a string, add it to the list
                elif isinstance(entity, dict): # If the element is a dictionary, check if it has a 'text' key
                    if 'text' in entity and isinstance(entity['text'], str): # If the key exists and the value is a string, add it to the list
                        texts.append(entity['text']) # If the key exists and the value is a string, add it to the list
            text = ' '.join(texts) # Concatenating all the 'text' from entities
        elif not isinstance(text, str):  # If not a string, convert it to an empty string
            text = ''
        messages_to_write.append([text, label]) # Adding the text and label to the list

    # Writing the text to the CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['text', 'label'])  # Writing the header
        writer.writerows(messages_to_write)

    print(
        f'Ready! The text from the JSON file was overwritten into a file {csv_filename}')


get_csv("data/text_spam_dataset", "result_not_spam.json", "not_spam.csv", 0)
get_csv("data/text_spam_dataset", "result_spam.json", "spam.csv", 1)
