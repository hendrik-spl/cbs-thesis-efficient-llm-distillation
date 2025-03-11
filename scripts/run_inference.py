import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import wandb
import argparse
from tqdm import tqdm
from datetime import datetime
from codecarbon import EmissionsTracker

from src.data.process_datasets import get_processed_hf_dataset
from src.models.model_utils import query_ollama
from src.prompts.sentiment import get_sentiment_prompt
from src.utils.clean_outputs import clean_llm_output_to_int
from src.utils.setup import ensure_dir_exists, set_seed
from src.evaluation.evaluate import evaluate_performance
from src.evaluation.eval_utils import get_duration

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with LLM models")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load (e.g., 'llama3.2:1b')")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'sentiment')")
    parser.add_argument("--limit", type=int, help="Limit the number of samples to process (default: 10)")
    parser.add_argument("--save_outputs", type=bool, default=True, help="Whether to save the outputs to a file")
    return parser.parse_args()

def run_inference(model_name: str, dataset: str, limit: int, save_outputs: str, wandb: wandb) -> str:
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

    sentences, true_labels = get_processed_hf_dataset("sentiment", split=False)

    if limit: 
        print(f"Limiting to {limit} samples.")
        sentences = sentences[:limit]
        true_labels = true_labels[:limit]

    prompts = [get_sentiment_prompt(sentences[i]) for i in range(len(sentences))]

    pred_labels = []
    results = {}
    results['data'] = []
    results['config'] = {
        'wandb_run_id': wandb.run.id,
        'model_name': model_name,
        'dataset': dataset,
        "timestamp" : timestamp,
    }

    emissions_output_dir = "results/metrics/emissions"
    ensure_dir_exists(emissions_output_dir)

    with EmissionsTracker(
        project_name="model-distillation",
        experiment_id=wandb.run.name,
        tracking_mode="process",
        output_dir=emissions_output_dir,
        log_level="warning"
        ) as tracker:
        for prompt in tqdm(prompts, total=len(prompts), desc=f"Running inference with {model_name} on {dataset}"):
            try:
                response = query_ollama(model_name, prompt)
                if response is None:
                    print(f"Warning: Received None response from model. Using default value.")
                    response = "1"
                pred_labels.append(response)
            except Exception as e:
                print(f"Error during inference: {e}")
                print("Using default value for this sample.")
                pred_labels.append("1")

    wandb.log({"inference_duration": get_duration(wandb.run.name)})
    wandb.log({"emissions": tracker.final_emissions})
    wandb.log({"energy_consumption": tracker._total_energy.kWh})

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
        inference_output_dir = f"models/{dataset}/teacher/{model_name}/inference_outputs"
        ensure_dir_exists(inference_output_dir)
        output_path = f"{inference_output_dir}/{wandb.run.name}_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        return output_path
    
    return None # No output path if not saving

def main():
    set_seed(42)
    args = parse_arguments()

    tags = [
        "test"
    ]

    config = {
        "model_name": args.model_name,
        "dataset": args.dataset
    }

    wandb.init(entity="cbs-thesis-efficient-llm-distillation", project="model-inference", tags=tags, config=config)

    output_path = run_inference(
                    model_name=args.model_name, 
                    dataset=args.dataset,
                    limit=args.limit,
                    save_outputs=args.save_outputs,
                    wandb=wandb)
    
    if output_path:
        evaluate_performance(results_path=output_path, dataset=args.dataset, wandb=wandb)

if __name__ == "__main__":
    main()