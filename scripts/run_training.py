import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import argparse
from codecarbon import EmissionsTracker
from trl import SFTConfig, SFTTrainer

from src.utils.setup import ensure_dir_exists, set_seed, ensure_cpu_in_codecarbon
from src.models.hf_utils import load_model_from_hf
from src.utils.logs import log_training_to_wandb
from datasets import load_from_disk
from src.data.data_transforms import DataTransforms

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run training with models")
    parser.add_argument("--student_model", type=str, required=True, help="Name of the model to load (e.g., 'google-t5:t5-small')")
    parser.add_argument("--teacher_model", type=str, required=True, help="Name of the teacher model to load (e.g., 'llama3.2:1b')")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'sentiment:50agree')")
    parser.add_argument("--inference_title", type=str, required=True, help="Title of the inference file to load")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train the model (default: 50)")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    return parser.parse_args()

def run_training(student_model: str, teacher_model: str, dataset_name: str, epochs: int, inference_title: str, wandb_run: wandb, learning_rate: float, batch_size: int):
    print(f"Running training with model {student_model} on dataset {dataset_name} from {teacher_model} for {epochs} epochs.")

    model, tokenizer, peft_config = load_model_from_hf(student_model, peft=False)
    dataset = load_from_disk(f"models/{dataset_name}/{teacher_model}/inference_outputs/{inference_title}")
    dataset = DataTransforms.split_data(dataset)

    model_output_dir = ensure_dir_exists(f"models/{dataset_name}/{student_model}/checkpoints/{wandb_run.name}")
    emissions_output_dir = ensure_dir_exists(f"results/metrics/emissions")

    training_args = SFTConfig(
        output_dir=model_output_dir,
        run_name=f"{student_model}_{dataset_name}_{wandb_run.name}",
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
        seed=42,
        packing=True,
        gradient_checkpointing=True,
        use_cache=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        peft_config=peft_config if peft_config else None
    )

    with EmissionsTracker(
        project_name="model-distillation",
        experiment_id=wandb_run.name,
        tracking_mode="machine",
        output_dir=emissions_output_dir,
        log_level="warning"
        ) as tracker:
        trainer.train()

    log_training_to_wandb(wandb_run, tracker, epochs)

    trainer.save_model(model_output_dir)

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
        "inference_title": args.inference_title,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate
    }

    wandb_run = wandb.init(entity="cbs-thesis-efficient-llm-distillation", project="model-training", tags=tags, config=config)

    run_training(
        student_model = args.student_model,
        teacher_model = args.teacher_model,
        dataset_name = args.dataset,
        epochs = args.epochs,
        inference_title = args.inference_title,
        wandb_run=wandb_run,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()