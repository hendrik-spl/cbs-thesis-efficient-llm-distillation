import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
from src.data.process_datasets import clean_input_text
from src.utils.setup import ensure_dir_exists

def load_data(version="50agree"):
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

def process_data(data):
    """
    Process the dataset by cleaning text and mapping labels.
    
    Args:
        data: The raw dataset to process.
        
    Returns:
        The processed dataset.
    """
    # Clean each sentence using the dataset's map method
    def clean_sentence(example):
        # Ensure input is a string
        if isinstance(example['sentence'], list):
            example['sentence'] = ''.join(example['sentence'])
        # Clean the text
        example['sentence'] = clean_input_text(example['sentence'])
        # Ensure output is a string
        if isinstance(example['sentence'], list):
            example['sentence'] = ''.join(example['sentence'])
        return example
    
    data = data.map(clean_sentence)
    
    def map_labels(example):
        example['label'] = label_mapping[example['label']]
        return example
    
    # Map numeric labels to text labels
    label_mapping = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    data = data.map(map_labels)
    
    return data

def split_data(data, test_size=0.3, val_test_ratio=0.5, seed=42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        data: The processed dataset to split.
        test_size: Proportion of data to reserve for validation and test.
        val_test_ratio: How to split the test_size portion between validation and test.
        seed: Random seed for reproducibility.
        
    Returns:
        A tuple of (train_data, val_data, test_data).
    """
    # Split the data into train (70%) and temp (30%)
    train_temp_split = data.train_test_split(test_size=test_size, seed=seed)
    train_data = train_temp_split['train']
    temp_data = train_temp_split['test']
    
    # Split the temp data into validation (15%) and test (15%)
    val_test_split = temp_data.train_test_split(test_size=val_test_ratio, seed=seed)
    val_data = val_test_split['train']
    test_data = val_test_split['test']
    
    return train_data, val_data, test_data

def save_data(train_data, val_data, test_data, dataset_path="data/sentiment"):
    """
    Save the datasets to disk.
    
    Args:
        train_data: The training dataset.
        val_data: The validation dataset.
        test_data: The test dataset.
        dataset_path: Path to save the datasets.
    """
    ensure_dir_exists(dataset_path)

    # Save the processed datasets
    train_data.save_to_disk(f"{dataset_path}/train")
    val_data.save_to_disk(f"{dataset_path}/val")
    test_data.save_to_disk(f"{dataset_path}/test")

def load_sentiment_from_hf(version="50agree"):
    """
    Loads the financial phrasebank dataset for sentiment analysis, processes it,
    splits it into train/val/test, and saves it to disk.

    Args:
        version: The version of the dataset to load. Default is "50agree". 
                Other options are "66agree", "75agree" and "allagree".
    """
    # Step 1: Load data
    raw_data = load_data(version)
    
    # Step 2: Process data
    processed_data = process_data(raw_data)
    
    # Step 3: Split data
    train_data, val_data, test_data = split_data(processed_data)
    
    # Step 4: Save data
    save_data(train_data, val_data, test_data)

if __name__ == "__main__":
    load_sentiment_from_hf()