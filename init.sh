#!/usr/bin/env bash
# init.sh

# Function to display info messages
function echo_info {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

function echo_error {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

function echo_success {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
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

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo_info "Script directory: $SCRIPT_DIR"

# Move to repository root for proper execution
cd "$SCRIPT_DIR/.." || exit 1
echo_info "Changed to directory: $(pwd)"

# Install uv if not already installed
if ! command -v uv > /dev/null 2>&1; then
    echo_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    # Verify uv installation
    if command -v uv > /dev/null 2>&1; then
        echo_success "uv installed successfully."
    else
        echo_error "uv installation failed. Please install manually and try again."
        exit 1
    fi
else
    echo_info "uv is already installed."
fi

# Clone the repository if not already cloned
REPO_DIR="cbs-thesis-efficient-llm-distillation"
REPO_URL="https://github.com/hendrik-spl/${REPO_DIR}.git"

if [ ! -d "$REPO_DIR" ]; then
    echo_info "Cloning repository from GitHub..."
    git clone "$REPO_URL"
    cd "$REPO_DIR" || exit 1
else
    echo_info "Repository already cloned. Pulling latest changes..."
    cd "$REPO_DIR" || exit 1
    if [[ "$RUNNING_ON_CLOUD" == "true" ]]; then
        echo_info "Running on UCloud. Resetting head..."
        git reset --hard HEAD
    fi
    git pull
fi

# Activate uv environment
echo_info "Current directory: $(pwd)"

# Quick fix: Use the copy mode to hide this warning:
# warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance. 
# If the cache and target directories are on different filesystems, hardlinking may not be supported.
# export UV_LINK_MODE=copy

# Set cache directory for uv
export UV_CACHE_DIR=$(pwd)/.cache/uv

# Check if .venv exists and remove if it's invalid
if [ -d ".venv" ] && [ ! -f ".venv/bin/python" ]; then
    echo_info "Found invalid .venv directory. Removing it..."
    rm -rf .venv
fi

# Create and activate virtual environment
echo_info "Creating virtual environment with uv..."
uv venv --python 3.11

# Verify virtual environment creation
if [ ! -f ".venv/bin/python" ]; then
    echo_error "Failed to create virtual environment. Python interpreter not found."
    exit 1
fi

echo_success "Virtual environment created successfully."

# Activate the virtual environment
echo_info "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with uv sync
echo_info "Installing dependencies with uv sync..."
uv pip install --upgrade pip setuptools wheel
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

# Set path where ollama models should be stored
export OLLAMA_MODELS=$(pwd)/.cache/ollama

# Install Ollama if not already installed
if ! command -v ollama > /dev/null 2>&1; then
    echo_info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Verify Ollama installation
    if command -v ollama > /dev/null 2>&1; then
        echo_success "Ollama installed successfully."
    else
        echo_error "Ollama installation failed. Please install manually."
    fi
else
    echo_info "Ollama is already installed."
fi

# Ensure Ollama cache directory exists
mkdir -p .cache/ollama

# Fix permissions for Ollama cache directory
sudo chown -R ollama: ollama -cache/ollama

# Check if Ollama is running before trying to start it
echo_info "Checking Ollama status..."
if lsof -i :11434 > /dev/null 2>&1 || nc -z localhost 11434 > /dev/null 2>&1; then
    echo_info "Ollama is already running."
else
    echo_info "Starting Ollama server..."
    ollama serve &
    # Wait a moment to ensure Ollama starts properly
    sleep 2
    # Check if it started successfully
    if lsof -i :11434 > /dev/null 2>&1 || nc -z localhost 11434 > /dev/null 2>&1; then
        echo_success "Ollama server started successfully."
    else
        echo_error "Failed to start Ollama server."
    fi
fi

echo_info "Initialization complete."