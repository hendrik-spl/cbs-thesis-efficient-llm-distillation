import wandb
import logging

from src.evaluation.eval_utils import get_duration

def log_gpu_info(wandb_run=None):
    """Get information about available NVIDIA GPUs."""
    gpu_count = 0
    gpu_names = []
    
    # Try using PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    except (ImportError, Exception) as e:
        print(f"Could not get GPU info via PyTorch: {e}")
    
    # Fallback to nvidia-smi if needed
    if gpu_count == 0:
        try:
            import subprocess
            result = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True)
            gpu_names = result.decode('utf-8').strip().split('\n')
            gpu_count = len(gpu_names)
        except Exception as e:
            print(f"Could not get GPU info via nvidia-smi: {e}")
    
    # Log GPU information to wandb if available
    if wandb_run:
        wandb_run.log({"gpu_count": gpu_count, 
                       "gpu_names": gpu_names if gpu_count > 0 else []
                       })

    return {
        "gpu_count": gpu_count,
        "gpu_names": gpu_names if gpu_count > 0 else []
    }

def log_inference_to_wandb(wandb: wandb, tracker, num_queries):
    wandb.log({"num_queries": num_queries})
    if tracker:
        total_duration = get_duration(wandb.name)
        wandb.log({"inference_duration [s]": total_duration})
        wandb.log({"energy_consumption [kWh]": tracker._total_energy.kWh})
        wandb.log({"emissions [CO₂eq, kg]": tracker.final_emissions})

def log_training_to_wandb(wandb: wandb, tracker):
    total_duration = get_duration(wandb.name)
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