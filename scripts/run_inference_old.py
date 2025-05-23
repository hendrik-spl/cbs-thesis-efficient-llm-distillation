import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import argparse
from tqdm import tqdm
from codecarbon import EmissionsTracker

from src.utils.setup import ensure_dir_exists, set_seed, ensure_cpu_in_codecarbon
from src.utils.logs import log_inference_to_wandb
from src.models.model_utils import query_with_sc, get_model_config
from src.evaluation.evaluate import evaluate_performance
from src.data.data_manager import SentimentDataManager

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with LLM models")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load (e.g., 'llama3.2:1b')")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'sentiment:50agree')")
    parser.add_argument("--limit", type=int, help="Limit the number of samples to process (default: 10)")
    parser.add_argument("--run_on_test", type=bool, default=False, help="Whether to run on the test set. If False, runs on the training set.")
    return parser.parse_args()

def run_inference(model_name: str, dataset_name: str, wandb_run: wandb, run_on_test: bool = False, limit: int = None, shots: int = 5) -> str:
    """
    Run inference with a given model on a dataset.

    Args:
        model_name: The name of the model to run inference with.
        dataset: The name of the dataset to run inference on.
        wandb: The wandb run to log metrics to.
        run_on_test: Whether to run on the test set. If False, runs on the training set.
        limit: The number of samples to process.
        shots: The number of samples to generate per prompt.

    Returns:
        None
    """
    print(f"Running inference with model {model_name} on dataset {dataset_name}. Limit: {limit}. Run on test: {run_on_test}. Shots: {shots}.")

    prompts, true_labels, pred_labels = SentimentDataManager.load_data(dataset_name=dataset_name, run_on_test=run_on_test, limit=limit)

    model_config, use_ollama = get_model_config(model_name)

    # manual override to load llama3.3:70b model from HF
    # from src.models.hf_utils import HF_Manager
    # model, tokenizer = HF_Manager.load_model(model_name="llama3.3:70b", peft=False)
    # model_config = (model, tokenizer)
    # use_ollama = False
    
    wandb_run.log({"use_ollama": use_ollama})

    with EmissionsTracker(
        project_name="model-distillation",
        experiment_id=wandb_run.name,
        tracking_mode="machine",
        output_dir=ensure_dir_exists("results/metrics/emissions"),
        log_level="warning"
        ) as tracker:
        for prompt in tqdm(prompts, total=len(prompts), desc=f"Running inference with {model_name} on {dataset_name}"):
            try:
                pred_labels.append(query_with_sc(
                    model=model_config,
                    prompt=prompt,
                    shots=shots,
                    use_ollama=use_ollama
                ))
            except Exception as e:
                print(f"Error during inference: {e}")

    log_inference_to_wandb(wandb_run, tracker, num_queries=len(prompts) * shots)
    
    SentimentDataManager.save_model_outputs(prompts, true_labels, pred_labels, dataset_name, model_name, wandb_run.name)

def main():
    set_seed(42)
    args = parse_arguments()
    ensure_cpu_in_codecarbon()    

    tags = [
        "test"
    ]

    config = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "limit": args.limit,
        "run_on_test": args.run_on_test,
    }

    custom_notes = ""

    wandb_run = wandb.init(entity="cbs-thesis-efficient-llm-distillation", project="model-inference-v2", tags=tags, config=config, notes=custom_notes)

    run_inference(
        model_name=args.model_name, 
        dataset_name=args.dataset,
        limit=args.limit,
        run_on_test=args.run_on_test,
        wandb_run=wandb_run,
        )
    
    evaluate_performance(args=args, wandb=wandb_run)

if __name__ == "__main__":
    main()