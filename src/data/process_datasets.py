from sklearn.model_selection import train_test_split

from src.data.load_datasets import load_sentiment

def get_processed_dataset(dataset, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and test sets.

    Args:
        dataset: The dataset to split. Available options: "sentiment".
        test_size: The proportion of the dataset to include in the test split. Default is 0.2.
        random_state: Controls the shuffling applied to the data before applying the split. Default is 42.

    Returns:
        Tuple: A tuple containing the training and test sets of the dataset. The first two elements are the training and test sentences, and the last two elements are the training and test labels.
    """
    if dataset == "sentiment":
        dataset = load_sentiment()
        train_sentences, test_sentences, train_labels, test_labels = train_test_split(dataset["sentence"], dataset["label"], test_size=test_size, random_state=random_state)
        return train_sentences, test_sentences, train_labels, test_labels
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Available options: 'sentiment'.")