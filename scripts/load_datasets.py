import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_manager import SentimentDataManager, GoldDataManager, SummaryManager
from src.data.data_transforms import DataTransforms

def load_sentiment(version="50agree"):
    dataset_path=f"data/sentiment:{version}"

    # Only continue if the directory does not exist yet
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}. Skipping download.")
        return

    # Step 1: Load data
    data = SentimentDataManager.load_original_data_hf(version)
    
    # Step 2: Process data
    data = SentimentDataManager.process_data(data)
    
    # Step 3: Split data
    data = DataTransforms.split_data(data)
    
    # Step 4: Save data
    DataTransforms.save_data(data, dataset_path)

    print(f"Successfully loaded and saved sentiment dataset at {dataset_path}.")

def load_gold():
    dataset_path=f"data/gold"

    # Only continue if the directory does not exist yet
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}. Skipping download.")
        return

    # Step 1: Load data
    data = GoldDataManager.load_original_data()
    
    # Step 2: Process data
    data = GoldDataManager.process_data(data)

    # Step 3: Split data
    data = DataTransforms.split_data(data)

    # Step 4: Save data
    DataTransforms.save_data(data, dataset_path)

    print(f"Successfully loaded and saved gold dataset at {dataset_path}.")

def load_summary():
    dataset_path=f"data/summary"

    # Only continue if the directory does not exist yet
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}. Skipping download.")
        return
    
    # Step 1: Load data
    data = SummaryManager.load_original_data()

    # Step 2: Process data
    data = SummaryManager.process_data(data)

    # Step 3: Save data
    DataTransforms.save_data(data, dataset_path)

    print(f"Successfully loaded and saved summary dataset at {dataset_path}.")

if __name__ == "__main__":
    load_sentiment()
    load_gold()
    load_summary()