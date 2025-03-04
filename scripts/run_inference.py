import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
from tqdm import tqdm
from datetime import datetime
from codecarbon import EmissionsTracker

from src.data.process_datasets import get_processed_dataset
from src.models.model_utils import query_ollama
from src.prompts.sentiment import get_sentiment_prompt
from src.utils.clean_outputs import clean_llm_output_to_int

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load a model and run queries")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--save_outputs", type=bool, default=True, help="Whether to save the outputs")
    return parser.parse_args()

def run_inference(model_name, dataset, save_outputs = True, limit=10):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Running inference with model {model_name} on dataset {dataset}.")

    sentences, true_labels = get_processed_dataset("sentiment", split=False)

    if limit: 
        print(f"Limiting to {limit} samples.")
        sentences = sentences[:limit]
        true_labels = true_labels[:limit]

    prompts = [get_sentiment_prompt(sentences[i]) for i in range(len(sentences))]

    pred_labels = []
    results = {}
    results['data'] = []
    results['config'] = {
        'model_name': model_name,
        'dataset': dataset,
        "timestamp" : timestamp
    }

    with EmissionsTracker(
        project_name="model-distillation",
        experiment_id="123",
        tracking_mode="process",
        output_dir="results/metrics/emissions",
        log_level="warning"
        ) as tracker:
        for prompt in tqdm(prompts):
            pred_labels.append(query_ollama(model_name, prompt))

    pred_labels = [clean_llm_output_to_int(label) for label in pred_labels]

    for i in range(len(sentences)):
        results['data'].append({
            "id" : i,
            "sentence" : sentences[i],
            "prompt" : prompts[i],
            "true_label" : true_labels[i],
            "pred_label" : pred_labels[i]
        })

    if save_outputs:
        output_path = f"data/inference_outputs/{model_name}/{dataset}_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(results, f)

def main():
    args = parse_arguments()

    run_inference(args.model_name, args.dataset, args.save_outputs)

if __name__ == "__main__":
    main()