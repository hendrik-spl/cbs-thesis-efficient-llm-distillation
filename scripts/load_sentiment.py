import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.sentiment_utils import load_sentiment_data_hf, process_sentiment_data
from src.data.process_datasets import split_data, save_data

def load_sentiment_from_hf(version="50agree"):
    """
    Loads the financial phrasebank dataset for sentiment analysis, processes it,
    splits it into train/val/test, and saves it to disk.

    Args:
        version: The version of the dataset to load. Default is "50agree". 
                Other options are "66agree", "75agree" and "allagree".
    """
    # Step 1: Load data
    raw_data = load_sentiment_data_hf(version)
    
    # Step 2: Process data
    data = process_sentiment_data(raw_data)
    
    # Step 3: Split data
    data = split_data(data)
    
    # Step 4: Save data
    save_data(data, dataset_path="data/sentiment")

if __name__ == "__main__":
    load_sentiment_from_hf()