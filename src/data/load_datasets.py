from datasets import load_dataset

def load_sentiment(version="sentences_50agree"):
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

# def load_text_classification():
#     pass

# def load_text_summarization():
#     pass