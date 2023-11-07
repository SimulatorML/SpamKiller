import openai
import yaml
import pandas as pd
import numpy as np
from textwrap import dedent
from config import OPENAI_API_KEY
from loguru import logger


class GptSpamClassifier:
    def __init__(self, api_key: str):
        """Initialize the classifier with an OpenAI API key."""
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

        with open("./config.yml", "r") as f:
            config = yaml.safe_load(f)
            self.path_not_spam_id = config["path_not_spam_id"]

        self.not_spam_ids = pd.read_csv(self.path_not_spam_id, sep=";")[
            "not_spam_id"
        ].tolist()

        logger.info("Initialized GptClassifier")

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Predicts if the message is spam or not.

        Parameters:
            X (pandas DataFrame): The input data to predict spam/not-spam for.

        Returns:
            dict: A dictionary containing label, reasons for the answer and tokens spent on request.
        """
        # Validate the input DataFrame to have the required columns
        logger.info("Predicting...")

        if not all(column in X for column in ['text', 'bio', 'from_id']):
            logger.error("Input DataFrame does not contain required columns: 'text', 'bio', 'from_id'.")
            return {'label': None, 'reasons': "Input is missing required columns.", 'tokens': 0}
        
        text = X['text'].iloc[0]
        bio = X['bio'].iloc[0]
        from_id = X['from_id'].iloc[0]
        spam_id = "no" if from_id in self.not_spam_ids else "no info"

        prompt = self._create_prompt(text, bio, spam_id)
        try:
            logger.info("Sending request to OpenAI...")

            # Call the OpenAI API to get a response
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": f"{prompt}"}]
            )

            # Interpret the API response
            response_text = response.choices[0].message.content
            tokens = response.usage.completion_tokens

            label, reasons = self._process_response(response_text)

            logger.info("Succesfully received response from OpenAI")
            return {'label': label, 'reasons': reasons, 'tokens': tokens}

        except Exception as e:
            logger.exception("An error occurred while predicting spam: %s", e)
            return {'label': None, 'reasons': f'An error occurred: {e}', 'tokens': 0}
        
    @staticmethod
    def _create_prompt(message: str, bio: str = None, spam_id: str = None) -> str:
        """Create a prompt for the GPT model to classify the message."""
        prompt = dedent(f""" \
            Is this message a spam? Also, consider user's bio and feature: user has sent spam-messages earlier - {spam_id}
            Reply only with tag <spam> if the message is spam, else <ham>. Second line must start with "Reasons:". Third and further line represent a short numeric list of few reasons why this decision
            <bio>
            {bio}
            </bio>
            <message>
            {message}
            </message>
            """)
        
        return prompt
    
    @staticmethod
    def _process_response(response: str) -> tuple[int, str]:
        """Process the response from OpenAI and extract the label and reasons."""
        lines = response.strip().split("\n")
        if not lines:
            return None, None
        
         # Determine the label based on the first line
        label = 1 if "<spam>" in lines[0] else 0 if "<ham>" in lines[0] else None

        # Join all lines except the first to form the reasons string
        reasons = "\n".join(lines[1:])

        return label, reasons
