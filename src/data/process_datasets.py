import re
from datasets import DatasetDict

from src.utils.setup import ensure_dir_exists

def clean_input_text(sentence):
    """
    Clean special characters from sentences.

    Args:
        sentence: The sentence to clean.

    Returns:
        The cleaned sentence.
    """
    # Clean special characters from sentences
    return re.sub(r'[^\x00-\x7F]+', '', sentence)

def clean_sentence(example):
    """
    Clean the sentence in the example.
    
    Args:
        example: The example to clean.
        
    Returns:
        The cleaned example.
    """
    if isinstance(example['sentence'], list):
        example['sentence'] = ''.join(example['sentence'])
    example['sentence'] = clean_input_text(example['sentence'])
    if isinstance(example['sentence'], list):
        example['sentence'] = ''.join(example['sentence'])
    return example

def map_labels(example):
    """
    Map the labels to human-readable form.

    Args:
        example: The example to map the labels for.
    
    Returns:
        The example with the mapped labels.
    """
    label_mapping = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    example['label'] = label_mapping[example['label']]
    return example

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

    merged_dataset = DatasetDict({
        'train': train_data,
        'valid': val_data,
        'test': test_data
    })
    
    return merged_dataset

def save_data(data, dataset_path):
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
    data.save_to_disk(f"{dataset_path}")