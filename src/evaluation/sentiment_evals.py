import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score

def measure_performance_sentiment_inference(true_labels, pred_labels, wandb_run):
    # Convert all labels to integers to ensure type consistency
    mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    
    # Transform labels from strings to integers if they are strings
    true_labels = [mapping.get(label, label) if isinstance(label, str) else label for label in true_labels]

    # Process predicted labels, handling -1 values
    processed_pred_labels = []
    for i, pred in enumerate(pred_labels):
        if pred == -1:
            # For invalid predictions (-1), assign a value guaranteed to be incorrect
            # Get the true label for this example
            true_label = true_labels[i]
            # Assign a value that's not the true label
            possible_values = [0, 1, 2]
            possible_values.remove(true_label)
            # Randomly select an incorrect value
            processed_pred_labels.append(possible_values[0])
        else:
            # For valid predictions, convert to integer if it's a string
            processed_pred_labels.append(mapping.get(pred, pred) if isinstance(pred, str) else pred)

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