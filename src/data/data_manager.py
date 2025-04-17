import os
import json
from datasets import load_dataset, Dataset, load_from_disk

from src.data.data_transforms import DataTransforms
from src.utils.setup import ensure_dir_exists
from src.prompts.sentiment import get_sentiment_prompt
from src.prompts.gold import get_gold_classification_prompt
from src.prompts.summary import get_summmary_prompt

def get_samples(dataset_name, limit: int = 5):
    if "sentiment" in dataset_name:
        prompts, _, _ = SentimentDataManager.load_data(dataset_name=dataset_name, run_on_test=True, limit=limit)
    elif "gold" in dataset_name:
        prompts, _, _ = GoldDataManager.load_data(dataset_name=dataset_name, run_on_test=True, limit=limit)
    return prompts

def load_data(dataset_name, run_on_test: bool = False, limit: int = None):
    if "sentiment" in dataset_name:
        prompts, true_labels, pred_labels = SentimentDataManager.load_data(dataset_name=dataset_name, run_on_test=run_on_test, limit=limit)
    elif "gold" in dataset_name:
        prompts, true_labels, pred_labels = GoldDataManager.load_data(dataset_name=dataset_name, run_on_test=run_on_test, limit=limit)
    elif "summary" in dataset_name:
        prompts, true_labels, pred_labels = SummaryManager.load_data(dataset_name=dataset_name, run_on_test=run_on_test, limit=limit)

    return prompts, true_labels, pred_labels

def save_model_outputs(prompts, true_labels, pred_labels, dataset_name, model_name, wandb_run_name):
    # convert pred_labels from list of dicts to list of strings so that it can be used as input for training
    if isinstance(pred_labels[0], dict):
        pred_labels_strings = [json.dumps(pred_label, indent=None, separators=(',', ':')) for pred_label in pred_labels]

    data = {
        "prompt": prompts,
        "true_label": true_labels,
        "completion": pred_labels_strings
    }
    dataset = Dataset.from_dict(data)
    output_dir = f"distillation-data/{dataset_name}/{model_name}/{wandb_run_name}"
    dataset.save_to_disk(ensure_dir_exists(output_dir))

class SummaryManager:

    @staticmethod
    def load_original_data():
        dataset = load_dataset(path="mrSoul7766/ECTSum", trust_remote_code=True)
        return dataset
    
    @staticmethod
    def load_data(dataset_name, run_on_test = False, limit = None):
        dataset = load_from_disk(f"data/summary")
        if run_on_test:
            data_split = dataset['test']
        else:
            data_split = dataset['train']

        if limit:
            dataset_size = len(data_split)
            actual_limit = min(limit, dataset_size)
            data_split = data_split.select(range(actual_limit))

        texts = data_split['text']
        prompts = [get_summmary_prompt(texts[i]) for i in range(len(texts))]
        true_labels = data_split['summary']
        pred_labels = []

        return prompts, true_labels, pred_labels

class GoldDataManager:

    @staticmethod
    def load_original_data():
        """
        Load the original dataset from Hugging Face.
        """
        dataset_path = "data/gold_dataset.csv"
        if not os.path.exists("data/gold_dataset.csv"):
            print(f"Could not find the original dataset at data/gold_dataset.csv. Please download it from Kaggle first: https://www.kaggle.com/datasets/daittan/gold-commodity-news-and-dimensions/data.")
            raise FileNotFoundError("Original dataset not found. Please download it from Kaggle.")
        else:
            dataset = load_dataset(path="csv", data_files=dataset_path)
        return dataset
    
    @staticmethod
    def process_data(data):
        data = data.shuffle(seed=42)
        data = data.remove_columns(["Dates", "URL"])

        return data
    
    @staticmethod
    def load_data(dataset_name, run_on_test: bool = False, limit: int = None):
        dataset = load_from_disk(f"data/{dataset_name}")
        if run_on_test:
            data_split = dataset['test']
        else:
            data_split = dataset['train']
        
        if limit:
            dataset_size = len(data_split)
            actual_limit = min(limit, dataset_size)
            data_split = data_split.select(range(actual_limit))

        news = data_split['News']
        prompts = [get_gold_classification_prompt(news[i]) for i in range(len(news))]

        column_names_mapping = {
            'Price or Not': 'price_or_not',
            'Direction Up': 'price_up',
            'Direction Constant': 'price_const_stable',
            'Direction Down': 'price_down',
            'PastPrice': 'past_price_info',
            'FuturePrice': 'future_price_info',
            'PastNews': 'past_gen_info',
            'FutureNews': 'future_gen_info',
            'Asset Comparision': 'asset_comparison'
        }

        label_columns = [col for col in data_split.column_names if col not in ['News', 'URL', 'Dates']]
        true_labels = []

        for i in range(len(data_split)):  # For each sample, extract the labels into a dictionary
            sample_labels = {}
            for col in label_columns:
                if col in column_names_mapping:
                    sample_labels[column_names_mapping[col]] = data_split[i][col]
                else:
                    sample_labels[col] = data_split[i][col]
            true_labels.append(sample_labels)

        pred_labels = []

        return prompts, true_labels, pred_labels
            
class SentimentDataManager:
    """Class for managing sentiment analysis datasets"""
    
    @staticmethod
    def load_original_data_hf(version="50agree"):
        """
        Loads the financial phrasebank dataset for sentiment analysis from Hugging Face Datasets.
        
        Args:
            version: The version of the dataset to load. Default is "50agree".
            
        Returns:
            The main data split of the dataset.
        """        
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
    
    @staticmethod
    def process_data(data):
        """
        Process the dataset by cleaning text and mapping labels.
        
        Args:
            data: The raw dataset to process.
            
        Returns:
            The processed dataset.
        """
        data = data.shuffle(seed=42)
        data = data.map(DataTransforms.clean_sentence)
        data = data.map(DataTransforms.map_labels, batched=False)
        return data
    
    @staticmethod
    def load_data(dataset_name, run_on_test: bool = False, limit: str = None):
        dataset = load_from_disk(f"data/{dataset_name}")
        if run_on_test:
            data_split = dataset['test']
        else:
            data_split = dataset['train']
        sentences = data_split['sentence']
        true_labels = data_split['label']

        if limit:
            print(f"Limiting to {limit} samples.")
            sentences = sentences[:limit]
            true_labels = true_labels[:limit]

        prompts = [get_sentiment_prompt(sentences[i]) for i in range(len(sentences))]
        
        pred_labels = []

        return prompts, true_labels, pred_labels