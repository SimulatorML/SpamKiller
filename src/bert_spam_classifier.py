from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_and_process_data():
    # Upload datasets
    df_spam = pd.read_csv('data/text_spam_dataset/cleaned_spam.csv', sep=';')
    df_not_spam = pd.read_csv(
        'data/text_spam_dataset/cleaned_not_spam.csv', sep=';')

    # Combine datasets
    df = pd.concat([df_spam, df_not_spam])

    # Divide the data into training and test samples
    return train_test_split(df['text'], df['label'], test_size=0.4)


def train_model(train_texts, train_labels):
    # Initialize the tokenizer and the model
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = BertForSequenceClassification.from_pretrained(
        'DeepPavlov/rubert-base-cased', num_labels=2)

    # Tokenize texts
    print("Tokenizing texts...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)

    # Create a dataset
    train_dataset = TextDataset(train_encodings, list(train_labels))

    # Create training parameters
    training_args = TrainingArguments(
        output_dir='./models',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Create an instance of the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.save_model()
    print("Starting training...")


def test_model(test_texts, test_labels):
    # Load the tokenizer and the model
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = BertForSequenceClassification.from_pretrained(
        './models', num_labels=2)

    # Tokenize texts
    print("Tokenizing texts...")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Create a dataset
    test_dataset = TextDataset(test_encodings, test_labels)

    # Create training parameters
    training_args = TrainingArguments(
        output_dir='./models',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Create an instance of the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset
    )

    # Evaluation of the model based on test data
    print("Evaluating model...")
    eval_result = trainer.evaluate()

    # Output of evaluation results
    for key, value in eval_result.items():
        print(f"{key}: {value}")

    # Get predictions
    predictions = trainer.predict(test_dataset).predictions
    predicted_labels = np.argmax(predictions, axis=1)

    # Output of classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_labels))


# Load and process data
train_texts, test_texts, train_labels, test_labels = load_and_process_data()

# Training model
train_model(train_texts.tolist(), train_labels.tolist())

# Testing model
test_model(test_texts.tolist(), test_labels.tolist())
