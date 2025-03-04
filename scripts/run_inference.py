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
from src.utils.setup import ensure_dir_exists
from src.evaluation.evaluate import evaluate_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with LLM models")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load (e.g., 'llama3.2:1b')")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'sentiment')")
    parser.add_argument("--limit", type=int, default=10, help="Limit the number of samples to process (default: 10)")
    parser.add_argument("--save_outputs", type=bool, default=True, help="Whether to save the outputs to a file")
    return parser.parse_args()

def run_inference(model_name: str, dataset: str, limit: int, save_outputs: str):
    """
    Run inference with a given model on a dataset.

    Args:
        model_name (str): The name of the model to load
        dataset (str): The name of the dataset to run inference on
        limit (int): The number of samples to process
        save_outputs (bool): Whether to save the outputs to a file

    Returns:
        None
    """

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
        for prompt in tqdm(prompts, total=len(prompts), desc=f"Running inference with {model_name} on {dataset}"):
            try:
                pred_labels.append(query_ollama(model_name, prompt))
            except Exception as e:
                print(f"Error querying model: {e}")
                return # Exit early

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
        output_dir = f"data/inference_outputs/{model_name}"
        ensure_dir_exists(output_dir)
        output_path = f"{output_dir}/{dataset}_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return output_path

def main():
    args = parse_arguments()

    output_path = run_inference(
                    model_name=args.model_name, 
                    dataset=args.dataset,
                    limit=args.limit,
                    save_outputs=args.save_outputs)
    
    evaluate_model(results_path=output_path, dataset=args.dataset)

if __name__ == "__main__":
    main()