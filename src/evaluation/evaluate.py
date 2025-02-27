from src.evaluation.evaluate_performance import evaluate_performance_sentiment

def evaluate_model(model, dataset, runs=10):
    if dataset == "sentiment":
        evaluate_performance_sentiment(model, runs)
    else:
        print("Invalid dataset.")