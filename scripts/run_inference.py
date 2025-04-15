import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import argparse
from tqdm import tqdm
from codecarbon import EmissionsTracker

from src.utils.setup import ensure_dir_exists, set_seed, ensure_cpu_in_codecarbon
from src.utils.logs import log_inference_to_wandb, log_gpu_info
from src.models.ollama_utils import query_ollama_sc, check_if_ollama_model_exists, track_samples_ollama
from src.models.hf_utils import HF_Manager
from src.evaluation.evaluate import evaluate_performance
from src.data.data_manager import load_data, save_model_outputs

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with LLM models")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load (e.g., 'llama3.2:1b')")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'sentiment:50agree')")
    parser.add_argument("--limit", type=int, help="Limit the number of samples to process (default: 10)")
    parser.add_argument("--run_on_test", action="store_true", help="Whether to run on the test set. If not specified, runs on the training set.")
    parser.add_argument("--use_ollama", action="store_true", help="Whether to use Ollama. If not specified, uses HF to run the model.")
    return parser.parse_args()

def run_inference_ollama(model_name: str, dataset_name: str, wandb_run: wandb, run_on_test: bool = False, limit: int = None, shots: int = 5) -> str:
    print(f"Running inference on ollama with model {model_name} on dataset {dataset_name}. Limit: {limit}. Run on test: {run_on_test}. Shots: {shots}.")

    prompts, true_labels, pred_labels = load_data(dataset_name=dataset_name, run_on_test=run_on_test, limit=limit)

    with EmissionsTracker(project_name="model-distillation", experiment_id=wandb_run.name, tracking_mode="machine", output_dir=ensure_dir_exists("results/metrics/emissions"), log_level="warning") as tracker:
        verbose = False
        verbose_counter = 0
        for prompt in tqdm(prompts, total=len(prompts), desc=f"Running inference with {model_name} on {dataset_name}"):
            verbose = verbose_counter < 2
            try:
                pred_labels.append(query_ollama_sc(model=model_name, prompt=prompt, dataset_name=dataset_name, shots=shots, verbose=verbose))
                verbose_counter += 1
            except Exception as e:
                print(f"Error during inference: {e}")

    track_samples_ollama(model_name, dataset_name)

    return tracker, len(prompts) * shots, prompts, true_labels, pred_labels

def run_inference_hf(model_name: str, dataset_name: str, wandb_run: wandb, run_on_test: bool = False, limit: int = None, shots: int = 5) -> str:
    print(f"Running inference on HF with model {model_name} on dataset {dataset_name}. Limit: {limit}. Run on test: {run_on_test}. Shots: {shots}.")

    prompts, true_labels, pred_labels = load_data(dataset_name=dataset_name, run_on_test=run_on_test, limit=limit)

    if os.path.exists(os.path.join(model_name, "adapter_config.json")):
        model, tokenizer = HF_Manager.load_finetuned_adapter(model_name)
    else:
        model, tokenizer = HF_Manager.load_model(model_name, peft=False)

    with EmissionsTracker(project_name="model-distillation", experiment_id=wandb_run.name, tracking_mode="machine", output_dir=ensure_dir_exists("results/metrics/emissions"), log_level="warning") as tracker:
        verbose = False
        verbose_counter = 0
        for prompt in tqdm(prompts, total=len(prompts), desc=f"Running inference with {model_name} on {dataset_name}"):
            verbose = verbose_counter < 2
            try:
                pred_labels.append(HF_Manager.query_hf_sc(model=model, tokenizer=tokenizer, dataset_name=dataset_name, prompt=prompt, shots=shots, verbose=verbose))
                verbose_counter += 1
            except Exception as e:
                print(f"Error during inference: {e}")

    HF_Manager.track_samples_hf(model, tokenizer, dataset_name)

    return tracker, len(prompts) * shots, prompts, true_labels, pred_labels

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
        "use_ollama": str(args.use_ollama)
    }

    custom_notes = ""

    wandb_run = wandb.init(entity="cbs-thesis-efficient-llm-distillation", project="model-inference-v2", tags=tags, config=config, notes=custom_notes)
    log_gpu_info(wandb_run)

    if str(args.use_ollama) == "True":
        check_if_ollama_model_exists(args.model_name)
        tracker, num_queries, prompts, true_labels, pred_labels = run_inference_ollama(
            model_name=args.model_name, 
            dataset_name=args.dataset,
            limit=args.limit,
            run_on_test=args.run_on_test,
            wandb_run=wandb_run
            )
    else:
        tracker, num_queries, prompts, true_labels, pred_labels = run_inference_hf(
            model_name=args.model_name, 
            dataset_name=args.dataset,
            limit=args.limit,
            run_on_test=args.run_on_test,
            wandb_run=wandb_run
            )
    
    log_inference_to_wandb(wandb_run, tracker, num_queries)
    
    if str(args.run_on_test) == "False":
        save_model_outputs(prompts, true_labels, pred_labels, args.dataset, args.model_name, wandb_run.name)
    
    evaluate_performance(true_labels, pred_labels, args.dataset, wandb_run)

if __name__ == "__main__":
    main()