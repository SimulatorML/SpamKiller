import pandas as pd
import openai
from dotenv import load_dotenv
import os

load_dotenv()


def augment_spam(df, num_samples, output_file, output_folder=''):
    """
    Function for augmenting spam messages

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe with spam messages
    num_samples : int
        Number of new messages to generate
    output_file : str
        Name of the output file
    output_folder : str, optional
        Path to the output folder, by default ''

    Returns
    -------
    new_df : pandas.core.frame.DataFrame
        Dataframe with new spam messages

    """

    # Create a list of spam messages
    spam_messages = df[df['label'] == 1]['text'].tolist()

    # Create a list for new messages
    new_messages = []

    for message in spam_messages:
        # Create a prompt
        response = openai.Completion.create(
            # Use the Davinci engine
            engine="text-davinci-003",
            prompt=message,
            max_tokens=200,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=num_samples,
        )

        # Add new messages to the list
        for choice in response.choices:
            new_messages.append(choice.text.strip())

    # Create a new dataframe
    new_df = pd.DataFrame({
        'text': new_messages,
        'label': [1] * len(new_messages),
    })

    # If output_folder is None, save the file to the current directory
    if output_folder is None:
        output_path = output_file
    else:
        # Create a path to the output file
        output_path = os.path.join(output_folder, output_file)

    # Save the dataframe to a csv file
    new_df.to_csv(output_path, index=False, sep=';')

    return new_df


df = pd.read_csv('data/text_spam_dataset/cleaned_spam.csv', sep=';')

# Put your OpenAI API key here
api_key_open_ai = os.getenv("API_KEY_OPEN_AI")
openai.api_key = api_key_open_ai

# Create a path to the output folder
output_folder = 'data/text_spam_dataset/'

# Create a new dataframe
augmented_df = augment_spam(
    df, num_samples=3, output_file='augmented_spam.csv', output_folder=output_folder)

# Print the dataframe
print(augmented_df)
