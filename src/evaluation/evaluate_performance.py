import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data.load_results import load_model_outputs_from_json

def measure_performance_sentiment(results_path, wandb_run: wandb):
    """
    Load the results of a sentiment analysis model and display the confusion matrix.

    Args:
        results_path (str): The path to the results file to load
        wandb_run: The Weights & Biases object
        
    Returns:
        None
    """
    results = load_model_outputs_from_json(results_path)

    # Define the ordering of classes in reverse (Positive first)
    classes = [2, 1, 0]  # Positive, Neutral, Negative
    class_names = ['Positive', 'Neutral', 'Negative']

    true_labels = [results['data'][i]['true_label'] for i in range(len(results['data']))]
    pred_labels = [results['data'][i]['pred_label'] for i in range(len(results['data']))]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    confusion = confusion_matrix(true_labels, pred_labels, labels=classes)
    
    wandb_run.log({"accuracy": accuracy})
    wandb_run.log({"f1": f1})

    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(wandb_run.name)
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names)
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=0)
    plt.text(0.5, -0.11, f'Accuracy: {accuracy:.2f} | F1-Score: {f1:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=8)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    
    wandb_run.log({"confusion_matrix": wandb.Image(plt)})