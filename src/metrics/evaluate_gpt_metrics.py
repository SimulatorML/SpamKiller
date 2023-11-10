import asyncio
import yaml
import time
import pandas as pd
import numpy as np
from loguru import logger
from src.models.gpt_classifier_validation import GptSpamClassifierValidation
from src.metrics import calculate_recall, calculate_specificity, calculate_precision


# Defining prices for gpt-3.5-turbo
PRICING_INPUT = 0.0010 # $ for 1k tokens
PRICING_OUTPUT = 0.0020 # $ for 1k tokens


def load_data(yaml_key: str = "test_gpt_data_path") -> pd.DataFrame:
    """Loading the data"""
    with open("./config.yml", "r") as f:
        config = yaml.safe_load(f)
        path_test_data = config[yaml_key]

    data = pd.read_csv(path_test_data, sep=";", index_col=None)

    logger.info(f"Loaded the data")
    return data


def save_predicted_data(data: pd.DataFrame, file_path: str, file_name: str, sep: str = ';'):
    """Saving predicted data by a model"""
    data.to_csv(f"{file_path}/{file_name}", sep=sep, index=False)

    logger.info(f"Saved {file_path}/{file_name}")


def evaluate_metrics():
    """Evaluating metrics"""
    # Loading the data for train
    train_data = load_data(yaml_key="train_gpt_data_path")
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']

    # Loading the data for test
    test_data = load_data(yaml_key="test_gpt_data_path")
    test_data['text'] = test_data['text'].fillna('')
    test_data['bio'] = test_data['bio'].fillna('')
    
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    
    # Define model
    model = GptSpamClassifierValidation()
    
    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    start_time = time.time()
    prediction_time = 0

    max_rpm = 10_000
    max_tpm = 1_000_000
    batch = 2_000 # Leaving 500 tokens for computing 1 response (which is upper estimate)
    response = []
    for i in range(0, len(X_test) + 1, batch): # Making predictions in parts to avoid TPM limit (40_000 TPM)
        start_time = time.time()

        sub_X = X_test.iloc[i:i + batch, :]
        sub_response = asyncio.run(model.predict(sub_X))

        response.extend(sub_response)

        batch_time = time.time() - start_time
        prediction_time += batch_time

        logger.info("Successfully predicted the batch!")

        if i + batch <= len(X_test) - 1:
            # Stop for 60 seconds to avoid reaching TPM limit (70 for assurance)
            logger.info("Waiting for 1 minute to not reach TPM limit")
            time.sleep(70)
        else: # if reached last iteration
            logger.info("Successfully predicted all the batches!")

    prediction_time_minutes = round(prediction_time / 60, 2)

    all_pred_labels = [item['label'] for item in response] # Getting all the predicted and failed cases
    sucessfully_predicted = [True if item in [0, 1] else False for item in all_pred_labels] # Getting mask of those cases that were successfully predicted
    y_true = np.array(y_test[sucessfully_predicted]) # Making y_true for cases OpenAI managed to predict and not fail
    y_pred = np.array([item['label'] for item in response if item['label'] is not None]) # Making y_pred for cases OpenAI managed to predict and not fail

    prompt_tokens = np.array([item['prompt_tokens'] for item in response if item['label'] is not None]) # If not error
    completion_tokens = np.array([item['completion_tokens'] for item in response if item['label'] is not None])
    time_spent = np.array([item['time_spent'] for item in response if item['label'] is not None])
    messages_quantity = len(X_test)

    logger.info(f"Messages quantity: {messages_quantity}")
    logger.info(f"Successfully managed to make a prediction for {len(y_pred)}/{messages_quantity} (all failed most likely reached TimeLimit)")
    logger.info(f"Time spent on processing {messages_quantity} messages: {prediction_time_minutes} minutes")

    # Save predictions to file
    data = pd.DataFrame(response)
    save_predicted_data(data=data, file_path='data/dataset', file_name='pred_test_gpt.csv')
    
    # Calculate offline-metrics
    recall = calculate_recall(y_true, y_pred)
    specificity = calculate_specificity(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)

    logger.info(f"Recall: {round(recall, 5)}")
    logger.info(f"Specificity: {round(specificity, 5)}")
    logger.info(f"Precision: {round(precision, 5)}")

    # Calculate business-metrics
    not_spam_mask = y_pred == 0
    spam_mask = y_pred == 1

    money_for_inputs = prompt_tokens * (PRICING_INPUT / 1000)
    money_for_outputs = completion_tokens * (PRICING_OUTPUT / 1000)
    total_worth = round(money_for_inputs.sum() + money_for_outputs.sum(), 4)

    logger.info(f"Money for inputs: {round(money_for_inputs.sum(), 4)}")
    logger.info(f"Money for outputs: {round(money_for_outputs.sum(), 4)}")
    logger.info(f"Total worth: {total_worth}")

    mean_worth_for_input = round(np.mean(money_for_inputs), 6)
    mean_worth_for_output = round(np.mean(money_for_outputs), 6)
    mean_worth_for_output = round(np.mean(money_for_outputs), 6)
    mean_worth_for_output_spam = round(np.mean(money_for_outputs[spam_mask]), 6)
    mean_worth_per_prediction = round(mean_worth_for_input + mean_worth_for_output, 6)

    logger.info(f"Mean money per input: {mean_worth_for_input}")
    logger.info(f"Mean money per output: {mean_worth_for_output}")
    logger.info(f"Mean money per output for spam-messages: {mean_worth_for_output_spam}")
    logger.info(f"Mean worth per prediction: {mean_worth_per_prediction}")

    mean_latency = round(np.mean(time_spent), 2) # Mean time spent on 1 prediction (seconds)
    mean_latency_not_spam = round(np.mean(time_spent[not_spam_mask]), 2)
    mean_latency_spam = round(np.mean(time_spent[spam_mask]), 2)
    qps = round(messages_quantity / prediction_time, 4)    # queries-per-second

    logger.info(f"Mean latency (time spent on 1 prediction): {mean_latency} seconds")
    logger.info(f"Mean latency for not-spam messages (time spent on 1 prediction): {mean_latency_not_spam} seconds")
    logger.info(f"Mean latency for spam messages (time spent on 1 prediction): {mean_latency_spam} seconds")
    logger.info(f"QPS (queries-per-second): {qps}")



if __name__ == "__main__":
    evaluate_metrics()