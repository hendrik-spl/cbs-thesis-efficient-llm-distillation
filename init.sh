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

# cd "$REPO_DIR"

# Activate uv environment
echo_info "Activating uv environment and installing dependencies..."
uv venv .venv
uv sync

# Install tensorflow dependent on environment
if [[ "$RUNNING_ON_CLOUD" == "true" ]]; then
    if command -v nvidia-smi > /dev/null 2>&1; then
        echo_info "Installing tf for Linux UCloud environment with cuda support..."
        uv add "tensorflow[and-cuda]"
    else
        echo_info "Installing tensorflow for CPU-only support..."
        uv add tensorflow
    fi
else
    echo_info "Installing tf for local MacOS environment..."
    uv add tensorflow-macos
fi

# Ensuring reproducibility
echo_info "Setting environment variable for reproducibility..."
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

echo_info "Initialization complete."