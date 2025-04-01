import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import argparse
from codecarbon import EmissionsTracker
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback

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

    # Count the number per class in the dataset
    from collections import Counter
    print(f"Train dataset split: {Counter(dataset['train']['true_label'])}")
    print(f"Test dataset split: {Counter(dataset['test']['true_label'])}")

    model_output_dir = ensure_dir_exists(f"models/{dataset_name}/{student_model}/checkpoints/{wandb_run.name}")
    emissions_output_dir = ensure_dir_exists(f"results/metrics/emissions")

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    training_args = SFTConfig(
        output_dir=model_output_dir,
        run_name=f"{student_model}_{dataset_name}_{wandb_run.name}",
        report_to='wandb',

        # Training schedule
        num_train_epochs=epochs, # Number of epochs to train
        learning_rate=learning_rate, # Learning rate
        lr_scheduler_type="cosine", # learning rate scheduler
        warmup_ratio=0.05, # warmup ratio for the learning rate scheduler      

        # Batch handling
        per_device_train_batch_size=batch_size, # Training batch size
        per_device_eval_batch_size=batch_size, # Evaluation batch size
        gradient_accumulation_steps=8, # effective batch size = batch_size * gradient_accumulation_steps

        # Evaluation and saving
        eval_strategy="epoch", # means evaluate by steps
        logging_strategy="epoch", # means log by steps
        save_strategy="best", # saves the best model
        save_total_limit=1, # saves only the best model
        load_best_model_at_end=True, # loads the best model at the end
        metric_for_best_model="eval_loss", # metric to use to compare models
        greater_is_better=False, # lower is better for loss

        # Regularization
        weight_decay=0.01, # add l2 regularization to the model
        max_grad_norm=0.5, # max gradient norm for clipping
        
        # Performance
        max_seq_length=256, 
        seed=42,
        packing=True, # pack the inputs for efficiency
        gradient_checkpointing=True, # use gradient checkpointing to save memory
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        peft_config=peft_config if peft_config else None,
        callbacks=[early_stopping],
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
    tokenizer.save_pretrained(model_output_dir)

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