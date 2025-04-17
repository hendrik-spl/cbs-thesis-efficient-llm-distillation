import wandb
import evaluate
import numpy as np
import pandas as pd

def measure_performance_summary_inference(true_labels, pred_labels, wandb_run=None):
    """
    Evaluate summarization performance between true and predicted summaries.
    
    Args:
        true_labels (List[str]): List of reference/ground truth summaries
        pred_labels (List[str]): List of model-generated summaries to evaluate
        wandb_run: Optional Weights & Biases run object for logging metrics
        
    Returns:
        dict: Dictionary containing the evaluation metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Import necessary libraries for evaluation
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
    
    # Format references for the evaluate library format
    # For each prediction, we have exactly one reference
    references = [[label] for label in true_labels]
    
    # Calculate ROUGE scores
    print("Calculating ROUGE scores...")
    try:
        rouge_results = rouge.compute(
            predictions=pred_labels,
            references=references,
            use_stemmer=True
        )
        
        # Add ROUGE metrics to the metrics dictionary
        for metric_name, value in rouge_results.items():
            metrics[metric_name] = value
    except Exception as e:
        print(f"Error calculating ROUGE scores: {e}")
    
    # Calculate BERTScore
    print("Calculating BERTScore (this may take a while for large datasets)...")
    try:
        # For BERTScore, we need to provide the single reference for each prediction
        # Since our references are in the format [[ref1], [ref2], ...], we extract the first element
        bertscore_results = bertscore.compute(
            predictions=pred_labels,
            references=[ref[0] for ref in references],  # Extract single reference for each prediction
            lang="en",
            model_type=None,  # Uses the suggested model for the target language
            verbose=True
        )
        
        # Add BERTScore metrics
        metrics['bertscore_precision'] = np.mean(bertscore_results['precision'])
        metrics['bertscore_recall'] = np.mean(bertscore_results['recall'])
        metrics['bertscore_f1'] = np.mean(bertscore_results['f1'])
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        metrics['bertscore_precision'] = None
        metrics['bertscore_recall'] = None
        metrics['bertscore_f1'] = None
        metrics['bertscore_hash'] = None
    
    # Print summary of results
    print("\n=== Summary Evaluation Results ===")
    if 'rouge1' in metrics:
        print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    if 'rouge2' in metrics:
        print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    if 'bertscore_f1' in metrics and metrics['bertscore_f1'] is not None:
        print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
    
    # Log metrics to wandb if provided
    if wandb_run:
        wandb_run.log({
            "rouge1": metrics.get('rouge1', None),
            "rouge2": metrics.get('rouge2', None),
            "bertscore_f1": metrics.get('bertscore_f1', None)
        })
        
        # Create a summary table for wandb
        valid_metrics = {k: v for k, v in metrics.items() if v is not None}
        metrics_table = pd.DataFrame({
            'Metric': list(valid_metrics.keys()),
            'Value': list(valid_metrics.values())
        })
        
        wandb_run.log({"metrics_summary": wandb.Table(dataframe=metrics_table)})
