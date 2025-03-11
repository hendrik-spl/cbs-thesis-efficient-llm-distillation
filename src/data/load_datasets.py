import torch
from datasets import load_dataset
from src.data.load_results import load_model_outputs_from_json
from src.data.pt_classes import SentimentDataSet

def load_sentiment_from_hf(version="sentences_50agree"):
    """
    Loads the financial phrasebank dataset for sentiment analysis from Hugging Face Datasets.
    Labels are integers from 0 to 2 where 0 is negative, 1 is neutral and 2 is positive.

    Args:
        version: The version of the dataset to load. Default is "sentences_50agree". Other options are "sentences_66agree", "sentences_75agree" and "sentences_allagree".

    Returns:
        datasets.Dataset: A Hugging Face Dataset object containing the training set of the financial phrasebank dataset for sentiment analysis.
    """
    dataset = load_dataset(path="takala/financial_phrasebank", name=version, trust_remote_code=True)

    return dataset['train']

def load_sentiment_dataset_from_json(model_name, file_name, tokenizer, split_size=0.8):
    path = f"data/inference_outputs/{file_name}"
    dataset = load_model_outputs_from_json(path)
    sentences = [d['sentence'] for d in dataset['data']]
    labels = [d['pred_label'] for d in dataset['data']]
    train_size = int(split_size * len(sentences))
    valid_size = len(sentences) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(SentimentDataSet(sentences, labels, tokenizer, 128), [train_size, valid_size])
    return train_dataset, valid_dataset