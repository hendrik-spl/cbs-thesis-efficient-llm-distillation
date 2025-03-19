import os
import csv
import torch
import keras
import random
import platform
import numpy as np
import tensorflow as tf

def get_root_dir() -> str:
    """
    Returns the root directory of the repository.

    Returns:
        str: The root directory of the
    """
    current_dir = os.getcwd()
    
    # Traverse up the directory tree until you find the root directory of the repo
    while not os.path.exists(os.path.join(current_dir, '.git')):
        current_dir = os.path.dirname(current_dir)

    return current_dir

def ensure_dir_exists(directory: str) -> None:
    """
    Ensures that the specified directory exists. If it does not, it creates it.

    Args:
        directory (str): The directory to ensure exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for reproducibility.

    Args:
        seed (int): The seed to use for reproducibility. Default is 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed

def setup_gpu() -> None:
    """
    Configures TensorFlow to use the GPU if available.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if platform.system() == 'Darwin':  # macOS
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("Configured TensorFlow to use Metal on macOS.")
            else:  # Assume CUDA for other platforms
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("Configured TensorFlow to use CUDA.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, using CPU.")

def ensure_cpu_in_codecarbon(cpu_model: str = "AMD EPYC 9454", power_consumption: int = 290) -> None:
    """
    Ensures that a specific CPU model exists in the CodeCarbon CPU power database.
    If it doesn't exist, adds it to the database.

    Args:
        cpu_model (str): The CPU model to add. Default is "AMD EPYC 9454".
        power_consumption (int): The power consumption in watts. Default is 290.
    """
    # Try different possible paths for the codecarbon CPU power database
    possible_paths = [
        "/.venv/lib/python3.11/site-packages/codecarbon/data/hardware/cpu_power.csv",
        os.path.join(os.path.expanduser("~"), ".venv/lib/python3.11/site-packages/codecarbon/data/hardware/cpu_power.csv"),
        os.path.join(get_root_dir(), ".venv/lib/python3.11/site-packages/codecarbon/data/hardware/cpu_power.csv"),
    ]
    
    if 'VIRTUAL_ENV' in os.environ:
        possible_paths.append(os.path.join(os.environ['VIRTUAL_ENV'], 'lib/python3.11/site-packages/codecarbon/data/hardware/cpu_power.csv'))

    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if not csv_path:
        print("Warning: CodeCarbon CPU power database not found in any of the expected locations.")
        return
    
    # Check if the entry already exists
    entry_exists = False
    try:
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0] == cpu_model:
                    entry_exists = True
                    print(f"CPU model {cpu_model} already exists in the database.")
                    break
        
        # Add the entry if it doesn't exist
        if not entry_exists:
            with open(csv_path, 'a') as file:
                file.write(f"{cpu_model},{power_consumption}")
            print(f"Added {cpu_model} with power consumption {power_consumption}W to the CodeCarbon CPU power database.")
    except Exception as e:
        print(f"Error updating CodeCarbon CPU power database: {e}")