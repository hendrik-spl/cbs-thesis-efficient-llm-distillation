import os
from datasets import load_dataset

def load_sentiment_dataset_from_json(model_name, dataset, file_name, test_size=0.2):
    path = f"models/{dataset}/{model_name}/inference_outputs/{file_name}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    
    dataset = load_dataset("json", data_files=path, field="data")
    if "pred_label" in dataset["train"][0]:
        dataset = dataset.rename_column("pred_label", "completion")
    dataset = dataset.remove_columns(["id", "sentence", "true_label"])
    dataset = dataset['train'].train_test_split(test_size, seed=42)

    return dataset