import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import argparse
import subprocess
from codecarbon import EmissionsTracker
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import EarlyStoppingCallback

from src.utils.setup import ensure_dir_exists, set_seed, ensure_cpu_in_codecarbon
from src.models.hf_utils import HF_Manager
from src.utils.logs import log_training_to_wandb, log_gpu_info
from datasets import load_from_disk
from src.data.data_transforms import DataTransforms
from scripts.training_config import get_sft_config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run training with models")
    parser.add_argument("--student_model", type=str, required=True, help="Name of the model to load")
    parser.add_argument("--teacher_model", type=str, required=True, help="Name of the teacher model to load")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--inference_title", type=str, required=True, help="Title of the inference file to load")
    parser.add_argument("--run_inference", type=bool, default=True, help="Whether to run inference after training (default: True)")
    return parser.parse_args()

def run_training(student_model: str, teacher_model: str, dataset_name: str, inference_wandb_title: str, wandb_run: wandb):
    print(f"Running training with model {student_model} on dataset {dataset_name} from {teacher_model}.")

    model, tokenizer = HF_Manager.load_model(student_model, peft=True)
    dataset = load_from_disk(f"distillation-data/{dataset_name}/{teacher_model}/{inference_wandb_title}") # load the distillation dataset 
    dataset = DataTransforms.split_data(dataset) # split the train dataset into train and test/valid sets

    model_output_dir = ensure_dir_exists(f"models/{dataset_name}/{student_model}/{wandb_run.name}")

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)

    if "sentiment" in dataset_name:
        response_template = "Final Label:"
    elif "gold" in dataset_name:
        response_template = "FINAL CLASSIFICATION:"
    elif "summary" in dataset_name:
        response_template = "FINAL SUMMARY OF YOUR TEXT:"
    data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    sft_config = get_sft_config(student_model, dataset_name, wandb_run, model_output_dir)
    training_args = SFTConfig(**sft_config)
 
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'], # test is the validation set of the distillation dataset
        tokenizer=tokenizer,
        callbacks=[early_stopping],
        data_collator=data_collator,
    )

    with EmissionsTracker(
        project_name="model-distillation",
        experiment_id=wandb_run.name,
        tracking_mode="machine",
        output_dir=ensure_dir_exists(f"results/metrics/emissions"),
        log_level="warning"
        ) as tracker:
        trainer.train()

    log_training_to_wandb(wandb_run, tracker)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    test_dataset = load_from_disk(f"data/{dataset_name}/test")
    try:
        HF_Manager.predict(model_path=model_output_dir, dataset=test_dataset, dataset_name=dataset_name, wandb_run=wandb_run)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return model_output_dir

    return model_output_dir

def main():
    set_seed(42)
    args = parse_arguments()
    ensure_cpu_in_codecarbon()

    tags = [
        "test"
    ]

    config = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "dataset": args.dataset,
        "inference_title": args.inference_title,
        "run_inference": args.run_inference
    }

    wandb_run = wandb.init(entity="cbs-thesis-efficient-llm-distillation", project="model-training", tags=tags, config=config)
    log_gpu_info(wandb_run)

    model_output_dir = run_training(
        student_model = args.student_model,
        teacher_model = args.teacher_model,
        dataset_name = args.dataset,
        inference_wandb_title = args.inference_title,
        wandb_run=wandb_run,
        )
    
    if str(args.run_inference) == "True":
        print("Running inference on the test set...")

        inference_script_path = os.path.join(os.path.dirname(__file__), "run_inference.py")
        inference_command = [
            "python", inference_script_path,
            "--model_name", f"{model_output_dir}",
            "--dataset", args.dataset,
            "--run_on_test", "True"
        ]

        try:
            subprocess.run(inference_command, check=True)
            print("Inference on test set completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running inference: {e}")

if __name__ == "__main__":
    main()