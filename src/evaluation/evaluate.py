from src.evaluation.evaluate_performance import evaluate_performance_sentiment

def evaluate_model(model, dataset, runs=10):
    """
    Evaluate the performance of a model on a dataset.

    Parameters:
        model (str): The model to evaluate. For example 'llama3.2:1b'.
        dataset (str): The dataset to evaluate the model on.
        runs (int): The number of runs to evaluate the model on. Default is 10.

    Returns:
        None
    """

    if dataset == "sentiment":
        evaluate_performance_sentiment(model, runs)
    else:
        print("Invalid dataset.")