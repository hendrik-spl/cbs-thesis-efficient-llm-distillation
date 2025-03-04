from src.evaluation.evaluate_performance import measure_performance_sentiment

def evaluate_model(results_path, dataset):
    """
    Evaluate the performance of a model on a dataset.

    Args:
        results_path (str): The path to the results file
        dataset (str): The name of the dataset to evaluate the model on

    Returns:
        None
    """
    if dataset == "sentiment":
        measure_performance_sentiment(
            results_path=results_path
        )
    else:
        print("Invalid dataset.")