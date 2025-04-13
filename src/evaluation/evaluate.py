from src.evaluation.evaluate_performance import measure_performance_sentiment, measure_performance_gold

def evaluate_performance(args, wandb):
    """
    Evaluate the performance of a model on a dataset.

    Args:
        args: The arguments to use for evaluation
        wandb: The Weights & Biases object

    Returns:
        None
    """
    if "sentiment" in args.dataset:
        measure_performance_sentiment(
            args=args,
            wandb_run=wandb
        )
    elif "gold" in args.dataset:
        measure_performance_gold(
            args=args,
            wandb_run=wandb
        )
    else:
        print("Invalid dataset.")