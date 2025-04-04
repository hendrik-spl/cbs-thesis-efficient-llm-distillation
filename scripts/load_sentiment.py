import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_manager import SentimentDataManager 
from src.data.data_transforms import DataTransforms

def load_sentiment_from_hf(version="allagree"):
    """
    Loads the financial phrasebank dataset for sentiment analysis, processes it,
    splits it into train/val/test, and saves it to disk.

    Args:
        version: The version of the dataset to load. Default is "50agree". 
                Other options are "66agree", "75agree" and "allagree".
    """
    # Step 1: Load data
    data = SentimentDataManager.load_original_data_hf(version)
    
    # Step 2: Process data
    data = SentimentDataManager.process_data(data)
    
    # Step 3: Split data
    data = DataTransforms.split_data(data)
    
    # Step 4: Save data
    DataTransforms.save_data(data, dataset_path=f"data/sentiment:{version}")

if __name__ == "__main__":
    load_sentiment_from_hf()