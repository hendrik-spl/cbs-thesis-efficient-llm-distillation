from src.evaluation.evaluate_performance import measure_performance_sentiment

def evaluate_performance(results_path, dataset, wandb):
    """
    Evaluate the performance of a model on a dataset.

    Args:
        results_path (str): The path to the results file
        dataset (str): The name of the dataset to evaluate the model on
        wandb: The Weights & Biases object

    Returns:
        None
    """
    if "sentiment" in dataset:
        measure_performance_sentiment(
            results_path=results_path,
            wandb=wandb
        )
    else:
        print("Invalid dataset.")