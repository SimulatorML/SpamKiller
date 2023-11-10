import asyncio
import time
from functools import partial
from textwrap import dedent
from typing import Optional, Tuple
import pandas as pd
import yaml
from loguru import logger
import openai
from openai import OpenAIError
from config import OPENAI_API_KEY


class GptSpamClassifierValidation:
    def __init__(self, api_key: str = OPENAI_API_KEY):
        """Initialize the classifier with an OpenAI API key."""
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

        # Due to inability to get user_id when loading new data from Spam Monitoring channels, usernames are used instead in Validation model
        self.not_spam_ids = [] # list filles in when fitting the model

        with open("./prompts.yml", "r") as f:
            prompts = yaml.safe_load(f)
        
        self.prompt = prompts["spam_classification_prompt"]

        logger.info("Initialized GptClassifier")

    def fit(self, X: pd.DataFrame, y) -> None:
        """
        Fits the model to the training data.

        Parameters:
            X (pandas DataFrame): The input features of shape (n_samples, n_features).
            y (pandas DataFrame or numpy array): The target labels of shape (n_samples,).

        Returns:
            None
        """
        not_spam_mask = y == 0
        self.not_spam_ids = X[not_spam_mask].from_id.to_list()
        logger.info('The model was successfully fitted')
        return None

    async def predict(self, X: pd.DataFrame) -> dict:
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
            return {'label': None, 'reasons': "Input is missing required columns.", 'prompt_tokens': 0, 'completion_tokens': 0, 'time_spent': 0}
         
        # Create a task for each row in the DataFrame
        tasks = [self._predict_row(X.iloc[i]) for i in range(len(X))]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        logger.info('Succesfully predicted')

        return results
    
    async def _predict_row(self, row):
        start_time = time.time()
        time_spent = None

        text = row['text'][:600]
        bio = row['bio'][:100]
        from_id = row['from_id']
        has_chat_history = "yes" if from_id in self.not_spam_ids else "no"

        prompt = self._create_prompt(text, bio, has_chat_history)

        try:
            # Call the _api_call method with a timeout
            response = await asyncio.wait_for(self._api_call(prompt), timeout=15)

            # Interpret the API response
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            response_text = response.choices[0].message.content

            label, reasons = self._process_response(response_text)
            time_spent = round(time.time() - start_time, 1)

            if label is None:
                logger.error("Couldn't interpret the OpenAI response")
                return {'label': None, 'reasons': "Couldn't interpret the OpenAI response", 'prompt_tokens': 0, 'completion_tokens': 0, 'time_spent': time_spent}
            
            logger.debug("Succesfully received response from OpenAI")
            return {'label': label, 'reasons': reasons, 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'time_spent': time_spent}
        except asyncio.TimeoutError:
            # Handle the TimeoutError
            time_spent = round(time.time() - start_time, 1)
            logger.error("The prediction took too long and was aborted after 15 seconds.")
            return {'label': None, 'reasons': 'Prediction timed out.', 'prompt_tokens': 0, 'completion_tokens': 0, 'time_spent': time_spent}
        except OpenAIError as e:
            # Handle OpenAI API errors
            time_spent = round(time.time() - start_time, 1) if time_spent is None else time_spent
            logger.exception("An error occurred with the OpenAI API: %s", e)
            return {'label': None, 'reasons': f'An OpenAI API error occurred: {e}', 'prompt_tokens': 0, 'completion_tokens': 0, 'time_spent': time_spent}
        except Exception as e:
            # Handle other unforeseen errors
            time_spent = round(time.time() - start_time, 1) if time_spent is None else time_spent
            logger.exception("An unexpected error occurred: %s", e)
            return {'label': None, 'reasons': f'An unexpected error occurred: {e}', 'prompt_tokens': 0, 'completion_tokens': 0, 'time_spent': time_spent}
        
    def _create_prompt(self, message: str, bio: str = None, has_chat_history: str = None) -> str:
        """Create a prompt for the GPT model to classify the message."""
        prompt = self.prompt.format(message_text=message, profile_bio=bio, has_chat_history=has_chat_history)
        logger.debug(f"Created prompt: {prompt}")
        return prompt
    
    async def _api_call(self, prompt: str):
        """Call the OpenAI API to get a response."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,  # Executor, None uses the default executor (a new thread)
            partial(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": prompt}]
            )
        )
        return response
    
    @staticmethod
    def _process_response(response: str) -> Tuple[Optional[int], Optional[str]]:
        """Process the response from OpenAI and extract the label and reasons."""
        lines = response.strip().split("\n")
        if not lines:
            return None, None
        
         # Determine the label based on the first line
        label = 1 if "<spam>" in lines[0] else 0 if "<not-spam>" in lines[0] else None

        # Join all lines except the first to form the reasons string; Empty if label == 0
        reasons = "\n".join(lines[1:]) if label == 1 else ''

        return label, reasons
