from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from typing import List, Dict
import pandas as pd
import wandb

def measure_performance_gold_inference(true_labels: List[Dict[str, int]], pred_labels: List[Dict[str, int]], wandb_run = None) -> Dict[str, float]:
    """
    Evaluate multi-label classification performance.
    
    Args:
        true_labels: List of dictionaries containing true labels
        pred_labels: List of dictionaries containing predicted labels
        wandb_run: Optional Weights & Biases run object for logging
        
    Returns:
        Dictionary containing performance metrics
    """
    # Determine categories if not provided
    categories = list(true_labels[0].keys())
    
    # Initialize metric trackers
    metrics_per_category = {}
    all_true = []
    all_pred = []
    
    # Evaluate each category separately
    for category in categories:
        category_true = []
        category_pred = []
        
        for true_dict, pred_dict in zip(true_labels, pred_labels):
            # Skip if category missing in either dictionary
            if category not in true_dict or category not in pred_dict:
                continue
                
            true_val = true_dict[category]
            
            # Handle missing category in pred_dict by treating as incorrect
            if category not in pred_dict:
                # Force an incorrect prediction
                pred_val = 1 - true_val if true_val in (0, 1) else 0
            else:
                pred_val = pred_dict[category]
                
                # IMPORTANT CHANGE: Penalize missing predictions (-1) 
                # by replacing with a value guaranteed to be incorrect
                if pred_val == -1:
                    # Convert to opposite of true value to ensure it's wrong
                    pred_val = 1 - true_val if true_val in (0, 1) else 0

            category_true.append(true_val)
            category_pred.append(pred_val)
            
            # Add to overall evaluation
            all_true.append(true_val)
            all_pred.append(pred_val)
        
        # Calculate metrics for this category
        if category_true and category_pred:
            try:
                accuracy = accuracy_score(category_true, category_pred)
                # Only calculate balanced_accuracy if there are at least two classes
                unique_classes = len(set(category_true))
                if unique_classes > 1:
                    balanced_acc = balanced_accuracy_score(category_true, category_pred)
                else:
                    balanced_acc = accuracy  # Fall back to regular accuracy
                    
                f1_micro = f1_score(category_true, category_pred, average='micro')
                
                metrics_per_category[category] = {
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'f1_micro': f1_micro,
                    'samples': len(category_true)
                }
            except Exception as e:
                print(f"Error calculating metrics for category {category}: {e}")
    
    # Calculate overall metrics
    overall_metrics = {}
    if all_true and all_pred:
        overall_metrics['accuracy'] = accuracy_score(all_true, all_pred)
        overall_metrics['balanced_accuracy'] = balanced_accuracy_score(all_true, all_pred)
        overall_metrics['f1_micro'] = f1_score(all_true, all_pred, average='micro')
    
    # Print results
    print("\n=== Overall Metrics ===")
    print(f"Accuracy: {overall_metrics.get('accuracy', 0):.4f}")
    print(f"Balanced Accuracy: {overall_metrics.get('balanced_accuracy', 0):.4f}")
    print(f"F1 Micro: {overall_metrics.get('f1_micro', 0):.4f}")
    
    metrics_df = pd.DataFrame([
        {
            'Category': cat,
            'Accuracy': metrics['accuracy'],
            'Balanced Acc': metrics['balanced_accuracy'],
            'F1 Micro': metrics['f1_micro'],
            'Samples': metrics['samples']
        }
        for cat, metrics in metrics_per_category.items()
    ])
    print("\n=== Metrics per Category ===")
    print(metrics_df)
    
    if wandb_run:
        wandb_run.log({"accuracy": overall_metrics.get('accuracy', 0)})
        wandb_run.log({"balanced_accuracy": overall_metrics.get('balanced_accuracy', 0)})
        wandb_run.log({"f1_micro": overall_metrics.get('f1_micro', 0)})
        wandb_run.log({"metrics_df": wandb.Table(dataframe=metrics_df)})
