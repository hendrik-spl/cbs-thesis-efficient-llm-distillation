import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import argparse
from datetime import datetime
from codecarbon import EmissionsTracker
# from transformers import Trainer, TrainingArguments
from trl import SFTConfig, SFTTrainer

from src.utils.setup import ensure_dir_exists, set_seed, ensure_cpu_in_codecarbon
from src.models.hf_utils import load_model_from_hf
from src.data.load_datasets import load_sentiment_dataset_from_json
from src.utils.logs import log_training_to_wandb

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run training with models")
    parser.add_argument("--student_model", type=str, required=True, help="Name of the model to load (e.g., 'google-t5:t5-small')")
    parser.add_argument("--teacher_model", type=str, required=True, help="Name of the teacher model to load (e.g., 'llama3.2:1b')")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'sentiment:50agree')")
    parser.add_argument("--json_file_name", type=str, required=True, help="Name of the JSON file to load the dataset from")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train the model (default: 50)")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    return parser.parse_args()

def run_training(student_model: str, teacher_student: str, dataset: str, epochs: int, json_file_name: str, wandb_instance: wandb, learning_rate: float, batch_size: int):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Running training with model {student_model} on dataset {dataset} from {teacher_student} for {epochs} epochs.")

    model, tokenizer, peft_config = load_model_from_hf(student_model, peft=True)
    if "sentiment" in dataset:
        dataset = load_sentiment_dataset_from_json(teacher_student, dataset, json_file_name)

    model_output_dir = f"models/{dataset}/{student_model}/checkpoints/{wandb_instance.name}_{timestamp}"
    ensure_dir_exists(model_output_dir)
    emissions_output_dir = f"results/metrics/emissions"
    ensure_dir_exists(emissions_output_dir)

    training_args = SFTConfig(
        output_dir=model_output_dir,
        run_name=f"{student_model}_{dataset}_{timestamp}",
        report_to='wandb',
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="best",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=42
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        peft_config=peft_config
    )

    with EmissionsTracker(
        project_name="model-distillation",
        experiment_id=wandb_instance.name,
        tracking_mode="machine",
        output_dir=emissions_output_dir,
        log_level="warning"
        ) as tracker:
        trainer.train()

    log_training_to_wandb(wandb_instance, tracker, epochs)

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
        "epochs": args.epochs,
        "json_file_name": args.json_file_name,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate
    }

    wandb_run = wandb.init(entity="cbs-thesis-efficient-llm-distillation", project="model-training", tags=tags, config=config)

    run_training(
        student_model = args.student_model,
        teacher_student = args.teacher_model,
        dataset = args.dataset,
        epochs = args.epochs,
        json_file_name = args.json_file_name,
        wandb_instance=wandb_run,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()