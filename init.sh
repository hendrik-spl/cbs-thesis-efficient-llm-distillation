#!/usr/bin/env bash
# init.sh

# Function to display info messages
function echo_info {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

function echo_error {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Detect Operating System
OS="$(uname)"
echo_info "Detected Operating System: $OS"

# Function to install packages on Linux
install_linux_packages() {
    echo_info "Updating package lists..."
    sudo apt-get update

    echo_info "Installing Git and curl..."
    sudo apt-get install -y git curl
}

# Function to install packages on macOS
install_macos_packages() {
    echo_info "Updating Homebrew..."
    brew update

    echo_info "Installing Git and curl..."
    brew install git curl
}

# Install necessary packages based on OS
if [[ "$OS" == "Linux" ]]; then
    install_linux_packages
    RUNNING_ON_CLOUD=true
elif [[ "$OS" == "Darwin" ]]; then
    install_macos_packages
    RUNNING_ON_CLOUD=false
else
    echo_error "Unsupported Operating System: $OS"
    exit 1
fi

echo_info "Checking GPU availability..."
if command -v nvidia-smi > /dev/null 2>&1; then
    echo_info "NVIDIA GPU found:"
    nvidia-smi
else
    echo_info "No GPU detected or driver missing."
fi

# Move one level up for proper execution of git pull
cd ..

# Install uv if not already installed
if ! command -v uv > /dev/null 2>&1; then
    echo_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo_info "uv is already installed."
fi

# Clone the repository if not already cloned
REPO_DIR="cbs-thesis-efficient-llm-distillation"
REPO_URL="https://github.com/hendrik-spl/${REPO_DIR}.git"

if [ ! -d "$REPO_DIR" ]; then
    echo_info "Cloning repository from GitHub..."
    git clone "$REPO_URL"
else
    echo_info "Repository already cloned. Pulling latest changes..."
    cd "$REPO_DIR"
    if [[ "$RUNNING_ON_CLOUD" == "true" ]]; then
        echo_info "Running on UCloud. Resetting head..."
        git reset --hard HEAD
        git pull
    fi
fi

# Quick fix: Use the copy mode to hide this warning:
# warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance. 
# If the cache and target directories are on different filesystems, hardlinking may not be supported.
export UV_LINK_MODE=copy

# Activate uv environment
echo_info "Activating uv environment and installing dependencies..."
echo_info "Current directory: $(pwd)"

# Check if .venv exists and remove if it's invalid
if [ -d ".venv" ]; then
    if [ ! -f ".venv/bin/python" ] && [ ! -f ".venv/Scripts/python.exe" ]; then
        echo_info "Found invalid .venv directory. Removing it..."
        rm -rf .venv
    fi
fi

uv venv .venv
uv sync

# Source environment variables from .env file
echo_info "Loading credentials from .env file..."
if [ -f .env ]; then
  set -a
  source .env
  set +a
  echo_info "Credentials loaded successfully."
else
  echo_info "No .env file found. Create one with your WANDB_API_KEY and HF_TOKEN."
fi

# Ensuring reproducibility for TensorFlow
echo_info "Setting environment variable for reproducibility..."
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export TF_ENABLE_ONEDNN_OPTS=0
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Set tokenizers parallelism environment variable to avoid warnings
export TOKENIZERS_PARALLELISM="false"

# Install Ollama if not already installed
if ! command -v ollama > /dev/null 2>&1; then
    echo_info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo_info "Ollama is already installed."
fi

echo_info "Initialization complete."