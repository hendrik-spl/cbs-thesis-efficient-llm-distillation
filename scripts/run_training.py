import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set tokenizers parallelism environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb
import argparse
from datetime import datetime
from codecarbon import EmissionsTracker
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.utils.setup import ensure_dir_exists, set_seed
from src.models.model_utils import load_model_from_hf
from src.data.load_datasets import load_sentiment_dataset_from_json
from src.evaluation.eval_utils import get_duration

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run training with models")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load (e.g., 'google-t5/t5-small')")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'sentiment')")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train the model (default: 50)")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--json_file_name", type=str, required=True, help="Name of the JSON file to load the dataset from")
    return parser.parse_args()

def run_training(model_name: str, dataset: str, epochs: int, json_file_name: str, wandb_instance: wandb, learning_rate: float, batch_size: int):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Running training with model {model_name} on dataset {dataset} for {epochs} epochs.")

    model, tokenizer = load_model_from_hf(model_name, num_labels=3)
    if dataset == "sentiment":
        train_dataset, valid_dataset = load_sentiment_dataset_from_json(model_name, json_file_name, tokenizer)

    model_output_dir = f"results/checkpoints/{model_name}_{timestamp}"
    ensure_dir_exists(model_output_dir)
    emissions_output_dir = "results/metrics/emissions"
    ensure_dir_exists(emissions_output_dir)

    # TODO:
    # Check out videos about training with Hugging Face
    # Define other training input parameters

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        run_name=f"{model_name}_{dataset}_{timestamp}",
        report_to='wandb',
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    with EmissionsTracker(
        project_name="model-distillation",
        experiment_id=wandb.run.name,
        tracking_mode="process",
        output_dir=emissions_output_dir,
        log_level="warning"
        ) as tracker:
        trainer.train()

    wandb.log({"training_duration": get_duration(wandb.run.name)})
    wandb.log({"emissions": tracker.final_emissions})
    wandb.log({"energy_consumption": tracker._total_energy.kWh})

    # get best model

    # save the entire model

def main():
    set_seed(42)
    args = parse_arguments()

    tags = [
        "test"
    ]

    config = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "json_file_name": args.json_file_name,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate
    }

    wandb.init(entity="cbs-thesis-efficient-llm-distillation", project="model-training", tags=tags, config=config)

    run_training(
        model_name = args.model_name, 
        dataset = args.dataset,
        epochs = args.epochs,
        json_file_name = args.json_file_name,
        wandb_instance=wandb,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()