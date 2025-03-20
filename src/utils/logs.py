import wandb
import logging

from src.evaluation.eval_utils import get_duration

def log_inference_to_wandb(wandb: wandb, tracker, num_queries):
    total_duration = get_duration(wandb.name)
    # avg_query_duration = total_duration / num_queries
    wandb.log({"num_queries": num_queries})
    wandb.log({"inference_duration [s]": total_duration})
    wandb.log({"energy_consumption [kWh]": tracker._total_energy.kWh})
    wandb.log({"emissions [CO₂eq, kg]": tracker.final_emissions})

def log_training_to_wandb(wandb: wandb, tracker, epochs):
    total_duration = get_duration(wandb.name)
    # avg_epoch_duration = total_duration / epochs
    # wandb.log({"num_epochs": epochs})
    wandb.log({"training_duration [s]": total_duration})
    wandb.log({"energy_consumption [kWh]": tracker._total_energy.kWh})
    wandb.log({"emissions [CO₂eq, kg]": tracker.final_emissions})

def init_logging(log_level: str = 'INFO', log_target: str = "file", log_file: str = 'results/logs/logs.log'):
    """
    Set up logging configuration.

    Parameters:
    log_level (str): The default log level, can be DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_target (str): Whether to log to 'file', 'console' or both.
    log_file (str): The file path to log to, if log_to_file is True. Defaults to 'app.log'.
    """
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    handlers = []
    if log_target == 'file' or log_target == 'both':
        handlers.append(logging.FileHandler(log_file))
    if log_target == 'console' or log_target == 'both':
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO), # defaults to logging.INFO if invalid input
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force = True
        )
            
    logging.info(f"Logging setup complete. Level: {log_level}. File logging: {log_target}")