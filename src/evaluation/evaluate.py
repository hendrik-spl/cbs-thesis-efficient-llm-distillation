from src.evaluation.sentiment_evals import measure_performance_sentiment_inference
from src.evaluation.gold_evals import measure_performance_gold_inference

def evaluate_performance(true_labels, pred_labels, dataset_name, wandb):
    """
    Evaluate the performance of the model based on the dataset type.
    
    Args:
        true_labels (list): The true labels from the dataset.
        pred_labels (list): The predicted labels from the model.
        dataset_name (str): The name of the dataset used for evaluation.
        wandb (wandb.run): The Weights & Biases run object.
        
    Returns:
        None
    """
    if "sentiment" in dataset_name:
        measure_performance_sentiment_inference(
            true_labels=true_labels,
            pred_labels=pred_labels,
            wandb_run=wandb
        )
    elif "gold" in dataset_name:
        measure_performance_gold_inference(
            true_labels=true_labels,
            pred_labels=pred_labels,
            wandb_run=wandb
        )
    else:
        print("Invalid dataset for evaluation.")