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
from src.models.ollama_utils import query_ollama_sc, check_if_ollama_model_exists
from src.prompts.sentiment import get_sentiment_prompt
from src.utils.setup import ensure_dir_exists, set_seed, ensure_cpu_in_codecarbon
from src.utils.logs import log_inference_to_wandb
from src.evaluation.evaluate import evaluate_performance

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with LLM models")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load (e.g., 'llama3.2:1b')")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'sentiment:50agree')")
    parser.add_argument("--limit", type=int, help="Limit the number of samples to process (default: 10)")
    parser.add_argument("--save_outputs", type=bool, default=True, help="Whether to save the outputs to a file")
    return parser.parse_args()

def run_inference(model_name: str, dataset: str, limit: int, save_outputs: str, wandb: wandb, shots: int = 5) -> str:
    """
    Run inference with a given model on a dataset.

    Args:
        model_name (str): The name of the model to load
        dataset (str): The name of the dataset to run inference on
        limit (int): The number of samples to process
        save_outputs (bool): Whether to save the outputs to a file
        wandb (wandb): The Weights & Biases run object
        shots (int): The number of shots to use for self-consistency

    Returns:
        None
    """
    print(f"Running inference with model {model_name} on dataset {dataset}.")

    sentences, true_labels = get_processed_hf_dataset(dataset=dataset, split_mode="none")

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
        "timestamp" : datetime.now().strftime('%Y%m%d_%H%M%S'),
    }

    emissions_output_dir = "results/metrics/emissions"
    ensure_dir_exists(emissions_output_dir)

    check_if_ollama_model_exists(model_name)

    with EmissionsTracker(
        project_name="model-distillation",
        experiment_id=wandb.run.name,
        tracking_mode="machine",
        output_dir=emissions_output_dir,
        log_level="warning"
        ) as tracker:
        for prompt in tqdm(prompts, total=len(prompts), desc=f"Running inference with {model_name} on {dataset}"):
            try:
                response = query_ollama_sc(model_name, prompt, shots)
                if response is None:
                    print(f"Warning: Received None response from model for prompt: {prompt}")
                    pred_labels.append(None)
                else:
                    pred_labels.append(response)
            except Exception as e:
                print(f"Error during inference: {e}")

    log_inference_to_wandb(wandb, tracker, num_queries=len(prompts) * shots)

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
        output_path = f"{inference_output_dir}/{wandb.run.name}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        return output_path
    
    return None # No output path if not saving

def main():
    set_seed(42)
    args = parse_arguments()
    ensure_cpu_in_codecarbon()

    tags = [
        "test"
    ]

    config = {
        "model_name": args.model_name,
        "dataset": args.dataset
    }

    custom_notes = ""

    wandb_run = wandb.init(entity="cbs-thesis-efficient-llm-distillation", project="model-inference-v2", tags=tags, config=config, notes=custom_notes)

    output_path = run_inference(
                    model_name=args.model_name, 
                    dataset=args.dataset,
                    limit=args.limit,
                    save_outputs=args.save_outputs,
                    wandb=wandb_run)
    
    if output_path:
        evaluate_performance(results_path=output_path, dataset=args.dataset, wandb=wandb_run)

if __name__ == "__main__":
    main()