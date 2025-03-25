import os
from datasets import load_dataset

def load_sentiment_from_hf(version="50agree"):
    """
    Loads the financial phrasebank dataset for sentiment analysis from Hugging Face Datasets.
    Labels are integers from 0 to 2 where 0 is negative, 1 is neutral and 2 is positive.

    Args:
        version: The version of the dataset to load. Default is "sentences_50agree". Other options are "sentences_66agree", "sentences_75agree" and "sentences_allagree".

    Returns:
        datasets.Dataset: A Hugging Face Dataset object containing the training set of the financial phrasebank dataset for sentiment analysis.
    """
    if version not in ["50agree", "66agree", "75agree", "allagree"]:
        raise ValueError(f"Invalid version {version}. Please choose from '50agree', '66agree', '75agree' or 'allagree'.")
    
    name = f"sentences_{version}"
    dataset = load_dataset(path="takala/financial_phrasebank", name=name, trust_remote_code=True)
    print(f"Loaded dataset {name} with {len(dataset['train'])} training samples.")

    return dataset['train']

def load_sentiment_dataset_from_json(model_name, dataset, file_name, test_size=0.2):
    path = f"models/{dataset}/{model_name}/inference_outputs/{file_name}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    
    dataset = load_dataset("json", data_files=path, field="data")
    if "pred_label" in dataset["train"][0]:
        dataset = dataset.rename_column("pred_label", "completion")
    dataset = dataset.remove_columns(["id", "sentence", "true_label"])
    dataset = dataset['train'].train_test_split(test_size, seed=42)

    return dataset