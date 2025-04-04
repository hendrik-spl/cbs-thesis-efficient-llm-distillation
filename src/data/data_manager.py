from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

from src.data.data_transforms import DataTransforms
from src.utils.setup import ensure_dir_exists
from src.prompts.sentiment import get_sentiment_prompt

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
    
    @staticmethod
    def save_model_outputs(prompts, true_labels, pred_labels, dataset_name, model_name, run_name):
        data = {
            "prompt": prompts,
            "true_label": true_labels,
            "completion": pred_labels
        }
        dataset = Dataset.from_dict(data)
        output_dir = f"models/{dataset_name}/{model_name}/inference_outputs/{run_name}"
        ensure_dir_exists(output_dir)
        dataset.save_to_disk(output_dir)