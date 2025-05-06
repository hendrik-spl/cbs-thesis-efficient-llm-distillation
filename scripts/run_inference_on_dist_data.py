import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import argparse

from src.utils.setup import set_seed, ensure_cpu_in_codecarbon
from src.utils.logs import log_inference_to_wandb, log_gpu_info
from src.evaluation.evaluate import evaluate_performance

from datasets import load_from_disk

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with LLM models")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'sentiment:50agree')")
    parser.add_argument("--inference_path", type=str, required=True)
    return parser.parse_args()

def compare_student_teacher_labels(teacher_dataset_path, student_dataset_path):
    teacher_dataset = load_from_disk(teacher_dataset_path)
    student_dataset = load_from_disk(student_dataset_path)

    teacher_prompts = teacher_dataset["prompt"]
    student_prompts = student_dataset["prompt"]

    true_labels = teacher_dataset["completion"]
    pred_labels = student_dataset["completion"]

    assert len(true_labels) == len(pred_labels), "Teacher and student datasets have different lengths."

    for i in range(len(teacher_prompts)):
        if teacher_prompts[i] != student_prompts[i]:
            print(f"Mismatch of prompts at index {i}: Teacher: {teacher_prompts[i]}, Student: {student_prompts[i]}")

    return true_labels, pred_labels

def main():
    set_seed(42)
    args = parse_arguments()
    ensure_cpu_in_codecarbon()

    tags = [
        "test on dist data",
    ]

    config = {
        "dataset": args.dataset,
        "inference_path": args.inference_path,
    }

    custom_notes = ""

    wandb_run = wandb.init(entity="cbs-thesis-efficient-llm-distillation", project="model-inference-v3", tags=tags, config=config, notes=custom_notes)
    log_gpu_info(wandb_run)

    if "sentiment" in args.dataset:
        teacher_dataset_path = "distillation-data/sentiment:50agree/llama3.3:70b/giddy-star-218"

    student_dataset_path = f"distillation-data/{args.dataset}/models/{args.dataset}/{args.inference_path}"
    # e.g. distillation-data/sentiment:50agree/models/sentiment:50agree/opt:125m/genial-energy-127/different-darkness-215

    true_labels, pred_labels = compare_student_teacher_labels(teacher_dataset_path, student_dataset_path)

    log_inference_to_wandb(wandb_run, None, len(pred_labels))
    
    evaluate_performance(true_labels, pred_labels, args.dataset, wandb_run)

    # save_model_outputs(prompts, true_labels, pred_labels, args.dataset, args.model_name, wandb_run.name)

if __name__ == "__main__":
    main()