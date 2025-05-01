import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score

def measure_performance_sentiment_inference(true_labels, pred_labels, wandb_run):
    """
    Evaluate sentiment classification performance, properly handling invalid predictions.
    
    Args:
        true_labels: List of sentiment labels (strings or integers)
        pred_labels: List of predicted labels (integers, with -1 for invalid)
        wandb_run: Optional Weights & Biases run object for logging
    """
    # Convert any string labels to integers
    mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    
    # Transform true labels from strings to integers if they are strings
    true_labels = [mapping.get(label, label) if isinstance(label, str) else label for label in true_labels]
    
    # Process predicted labels, handling -1 values
    processed_pred_labels = []
    invalid_count = 0
    
    for i, pred in enumerate(pred_labels):
        if pred == -1:
            invalid_count += 1
            # For invalid predictions (-1), assign a value guaranteed to be incorrect
            true_label = true_labels[i]
            # Choose a label different from the true label
            incorrect_labels = [0, 1, 2]
            if true_label in incorrect_labels:
                incorrect_labels.remove(true_label)
            processed_pred_labels.append(incorrect_labels[0])
        else:
            # Already numeric, just append
            processed_pred_labels.append(pred)
    
    # Define the ordering of classes (Positive first for plot!)
    classes = [2, 1, 0]  # Positive, Neutral, Negative
    class_names = ['Positive', 'Neutral', 'Negative']

    # Calculate metrics
    accuracy = accuracy_score(true_labels, processed_pred_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, processed_pred_labels)
    f1_micro = f1_score(true_labels, processed_pred_labels, average='micro')
    confusion = confusion_matrix(true_labels, processed_pred_labels, labels=classes)

    # Calculate percentage of invalid predictions
    invalid_percentage = (invalid_count / len(pred_labels)) * 100 if pred_labels else 0
    
    # Log metrics to wandb
    wandb_run.log({
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "balanced_accuracy": balanced_accuracy,
        "invalid_predictions": invalid_count,
        "invalid_predictions_percentage": invalid_percentage
    })

    # Create and log confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(wandb_run.name)
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names)
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=0)
    plt.text(0.5, -0.11, f'Balanced Accuracy: {balanced_accuracy:.2f} | F1 micro: {f1_micro:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=8)
    plt.text(0.5, -0.15, f'Invalid predictions: {invalid_count} ({invalid_percentage:.1f}%)', ha='center', va='center', transform=plt.gca().transAxes, fontsize=8)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    
    wandb_run.log({"confusion_matrix": wandb.Image(plt)})
    
def measure_performance_sentiment_from_dataset(args, wandb_run: wandb):
    """
    Load the results of a sentiment analysis model and display the confusion matrix.

    Args:
        args: The arguments to use for evaluation
        wandb_run: The Weights & Biases object
        
    Returns:
        None
    """
    results = load_from_disk(f"distillation-data/{args.dataset}/{args.model_name}/{wandb_run.name}")

    mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    # Extract the true and predicted labels
    true_labels = [results[i]['true_label'] for i in range(len(results))]
    pred_labels = [results[i]['completion'] for i in range(len(results))]

    # Transform labels from strings to integers. Predicted labels are already integers.
    true_labels = [mapping[label] if isinstance(label, str) else label for label in true_labels]
    pred_labels = [mapping[label] if isinstance(label, str) else label for label in pred_labels]

    # Define the ordering of classes in reverse (Positive first for plot!)
    classes = [2, 1, 0]  # Positive, Neutral, Negative
    class_names = ['Positive', 'Neutral', 'Negative']
    
    accuracy = accuracy_score(true_labels, pred_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    f1_micro = f1_score(true_labels, pred_labels, average='micro')
    confusion = confusion_matrix(true_labels, pred_labels, labels=classes)
    
    wandb_run.log({"accuracy": accuracy})
    wandb_run.log({"f1_micro": f1_micro})
    wandb_run.log({"balanced_accuracy": balanced_accuracy})

    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(wandb_run.name)
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names)
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=0)
    plt.text(0.5, -0.11, f'Balanced Accuracy: {balanced_accuracy:.2f} | F1 micro: {f1_micro:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=8)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    
    wandb_run.log({"confusion_matrix": wandb.Image(plt)})