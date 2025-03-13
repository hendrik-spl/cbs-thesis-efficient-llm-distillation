import re
from sklearn.model_selection import train_test_split

from src.data.load_datasets import load_sentiment_from_hf

def clean_input_text(text):
    # Clean special characters from sentences
    return [re.sub(r'[^\x00-\x7F]+', '', sentence) for sentence in text]

def get_processed_hf_dataset(dataset, split_mode="train-valid-test", train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42):
    """
    Processes and splits the dataset into training, validation, and test sets.

    Args:
        dataset: The dataset to split. Available options: "sentiment".
        split_mode: Splitting mode. Options: "none" (no split), "train-test" (two-way split), 
                   "train-valid-test" (three-way split). Default is "train-valid-test".
        train_size: The proportion of the dataset for training. Default is 0.7.
        valid_size: The proportion of the dataset for validation. Default is 0.15.
        test_size: The proportion of the dataset for testing. Default is 0.15.
        random_state: Controls the shuffling applied to the data before applying the split. Default is 42.

    Returns:
        If split_mode is "none": Tuple (sentences, labels)
        If split_mode is "train-test": Tuple (train_sentences, test_sentences, train_labels, test_labels)
        If split_mode is "train-valid-test": Tuple (train_sentences, valid_sentences, test_sentences, 
                                                   train_labels, valid_labels, test_labels)
    """
    if dataset == "sentiment":
        hf_dataset = load_sentiment_from_hf()
        
        # Extract data from the Dataset object
        sentences = list(hf_dataset["sentence"])
        labels = list(hf_dataset["label"])
        
        # Clean the sentences
        sentences = clean_input_text(sentences)
        
        if split_mode == "none":
            return sentences, labels
        
        elif split_mode == "train-test":
            train_sentences, test_sentences, train_labels, test_labels = train_test_split(
                sentences, labels, test_size=test_size, random_state=random_state
            )
            return train_sentences, test_sentences, train_labels, test_labels
        
        elif split_mode == "train-valid-test":
            train_valid_sentences, test_sentences, train_valid_labels, test_labels = train_test_split(
                sentences, labels, test_size=test_size, random_state=random_state
            )
            
            valid_ratio = valid_size / (train_size + valid_size)
            
            train_sentences, valid_sentences, train_labels, valid_labels = train_test_split(
                train_valid_sentences, train_valid_labels, test_size=valid_ratio, random_state=random_state
            )
            
            return train_sentences, valid_sentences, test_sentences, train_labels, valid_labels, test_labels
        
        else:
            raise ValueError(f"Invalid split_mode: {split_mode}. Available options: 'none', 'train-test', 'train-valid-test'.")
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Available options: 'sentiment'.")