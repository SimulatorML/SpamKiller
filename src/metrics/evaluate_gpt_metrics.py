import asyncio
import yaml
import time
import pandas as pd
import numpy as np
from loguru import logger
from src.models import GptSpamClassifier
from src.metrics import calculate_recall, calculate_specificity, calculate_precision


# Defining prices for gpt-3.5-turbo
PRICING_INPUT = 0.0010 # $ for 1k tokens
PRICING_OUTPUT = 0.0020 # $ for 1k tokens


def load_data() -> pd.DataFrame:
    """Loading the data"""
    with open("./config.yml", "r") as f:
        config = yaml.safe_load(f)
        path_test_data = config["test_gpt_data_path"]

    data = pd.read_csv(path_test_data, sep=";", index_col=None)

    logger.info(f"Loaded the data")
    return data


def save_predicted_data(data: pd.DataFrame, file_path: str, file_name: str, sep: str = ';'):
    """Saving predicted data by a model"""
    data.to_csv(f"{file_path}/{file_name}", sep=sep, index=False)

    logger.info(f"Saved {file_path}/{file_name}")


def evaluate_metrics():
    """Evaluating metrics"""
    data = load_data()
    
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Define model
    model = GptSpamClassifier()
    
    # Make predictions
    start_time = time.time()

    response = asyncio.run(model.predict(X))

    prediction_time = time.time() - start_time
    prediction_time_minutes = round(prediction_time / 60, 2)

    pred_labels = [item['label'] for item in response]
    prompt_tokens = [item['prompt_tokens'] for item in response if item['label'] is not None] # If not error
    completion_tokens = [item['completion_tokens'] for item in response if item['label'] is not None]
    time_spent = [item['time_spent'] for item in response if item['label'] is not None]
    messages_quantity = len(X)

    logger.info(f"Messages quantity: {messages_quantity}")
    logger.info(f"Time spent on processing {messages_quantity} messages: {prediction_time_minutes} minutes")

    # Save predictions to file
    data = pd.DataFrame(response)
    save_predicted_data(data=data, file_path='data/dataset', file_name='pred_test_gpt.csv')
    
    # Calculate offline-metrics
    recall = calculate_recall(y, pred_labels)
    specificity = calculate_specificity(y, pred_labels)
    precision = calculate_precision(y, pred_labels)

    logger.info(f"Recall: {round(recall, 5)}")
    logger.info(f"Specificity: {round(specificity, 5)}")
    logger.info(f"Precision: {round(precision, 5)}")

    # Calculate business-metrics
    money_for_inputs = np.array(prompt_tokens) * (PRICING_INPUT / 1000)
    money_for_outputs = np.array(completion_tokens) * (PRICING_OUTPUT / 1000)
    total_worth = round(money_for_inputs.sum() + money_for_outputs.sum(), 4)

    logger.info(f"Money for inputs: {round(money_for_inputs.sum(), 4)}")
    logger.info(f"Money for outputs: {round(money_for_outputs.sum(), 4)}")
    logger.info(f"Total worth: {total_worth}")

    mean_worth_for_input = round(np.mean(money_for_inputs), 6)
    mean_worth_for_output = round(np.mean(money_for_outputs), 6)
    mean_worth_per_prediction = round(mean_worth_for_input + mean_worth_for_output, 6)

    logger.info(f"Mean money per input: {mean_worth_for_input}")
    logger.info(f"Mean money per output: {mean_worth_for_output}")
    logger.info(f"Mean worth per prediction: {mean_worth_per_prediction}")

    mean_latency = round(np.mean(time_spent), 2) # Mean time spent on 1 prediction (seconds)
    qps = round(messages_quantity / prediction_time, 4)    # queries-per-second

    logger.info(f"Mean latency (time spent on 1 prediction): {mean_latency} seconds")
    logger.info(f"QPS (queries-per-second): {qps}")



if __name__ == "__main__":
    evaluate_metrics()