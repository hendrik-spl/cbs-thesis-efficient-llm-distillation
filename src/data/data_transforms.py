import re
from datasets import DatasetDict, Dataset

from src.utils.setup import ensure_dir_exists

class DataTransforms:
    @staticmethod
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
    
    @staticmethod
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
        example['sentence'] = DataTransforms.clean_input_text(example['sentence'])
        if isinstance(example['sentence'], list):
            example['sentence'] = ''.join(example['sentence'])
        return example

    @staticmethod
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

        if isinstance(example['label'], str) and example['label'].isdigit():
            example['label'] = label_mapping[int(example['label'])]
        elif isinstance(example['label'], int):
            example['label'] = label_mapping[example['label']]

        return example

    @staticmethod
    def split_data(data: Dataset, test_size=0.2, seed=42):
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            data: The processed dataset to split.
            test_size: Proportion of data to reserve for validation and test.
            val_test_ratio: How to split the test_size portion between validation and test.
            seed: Random seed for reproducibility.
            
        Returns:
            The split dataset.
        """
        # Split the data into train (80%) and temp (20%)
        train_temp_split = data.train_test_split(test_size=test_size, seed=seed)
        train_data = train_temp_split['train']
        test_data = train_temp_split['test']
        
        merged_dataset = DatasetDict({
            'train': train_data,
            'test': test_data
        })
        
        return merged_dataset

    @staticmethod
    def save_data(data, dataset_path):
        """
        Save the datasets to disk.
        
        Args:
            data: The dataset to save.
            dataset_path: The path to save the dataset to.

        Returns:
            None
        """
        ensure_dir_exists(dataset_path)

        # Save the processed datasets
        data.save_to_disk(f"{dataset_path}")