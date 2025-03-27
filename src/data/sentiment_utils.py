import os
from datasets import load_dataset
from src.data.process_datasets import clean_sentence, map_labels

def load_sentiment_dataset_from_json(model_name, dataset, file_name, test_size=0.2):
    """
    Load the inference outputs from a model and dataset from a JSON file.

    Args:
        model_name: The name of the model.
        dataset: The name of the dataset.
        file_name: The name of the JSON file to load.
        test_size: The proportion of the dataset to use for testing. Default is 0.2.

    Returns:
        The train and test datasets
    """
    path = f"models/{dataset}/{model_name}/inference_outputs/{file_name}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    
    dataset = load_dataset("json", data_files=path, field="data")
    if "pred_label" in dataset["train"][0]:
        dataset = dataset.rename_column("pred_label", "completion")
    dataset = dataset.remove_columns(["id", "sentence", "true_label"])
    dataset = dataset['train'].train_test_split(test_size, seed=42)

    return dataset

def load_sentiment_data_hf(version="50agree"):
    """
    Loads the financial phrasebank dataset for sentiment analysis from Hugging Face Datasets.
    
    Args:
        version: The version of the dataset to load. Default is "50agree".
        
    Returns:
        The main data split of the dataset.
    """
    if version not in ["50agree", "66agree", "75agree", "allagree"]:
        raise ValueError(f"Invalid version {version}. Please choose from '50agree', '66agree', '75agree' or 'allagree'.")
    
    name = f"sentences_{version}"
    dataset = load_dataset(path="takala/financial_phrasebank", name=name, trust_remote_code=True)
    print(f"Loaded dataset {name} with {len(dataset['train'])} training samples.")

    # Try to get the main data split to work with
    if 'train' in dataset:
        data = dataset['train']
    else:
        # If no 'train' split, use the first available split
        main_split = list(dataset.keys())[0]
        data = dataset[main_split]
        
    return data

def process_sentiment_data(data):
    """
    Process the dataset by cleaning text and mapping labels.
    
    Args:
        data: The raw dataset to process.
        
    Returns:
        The processed dataset.
    """

    # Shuffle the data
    data = data.shuffle(seed=42)
    
    # Clean the sentences
    data = data.map(clean_sentence)
    
    # Map the labels from integers to strings
    data = data.map(map_labels)
    
    return data

