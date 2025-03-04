import os
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