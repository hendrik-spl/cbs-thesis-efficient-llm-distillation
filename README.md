# Resource-Efficient Large Language Model Distillation

This repository contains the code, models, and experiments for our Master's thesis titled **"Distilling Intelligence: Leveraging Knowledge Distillation for Improved Resource Efficiency of Large Language Models in Business Applications"**. The research focuses on making LLMs more sustainable through knowledge distillation while maintaining strong performance on financial analysis tasks. The full thesis is available upon request.

## Project Overview

This project investigates hard-label knowledge distillation as a technique to improve the resource efficiency of Large Language Models (LLMs) while preserving acceptable performance levels across varying analytical tasks. The research employs a quantitative, empirical approach to evaluate distilled models on three increasingly complex financial NLP tasks:

- **Sentiment Analysis** - Classifying financial phrases as positive, negative, or neutral
- **Text Classification** - Categorizing commodity market news headlines into multiple financial dimensions
- **Summarization** - Condensing lengthy earnings call transcripts into concise bullet points

Through these experiments, we empirically validate whether smaller, distilled models can achieve similar performance to larger teacher models while consuming significantly fewer compute resources, energy, and costs.

## Relevant Parts of the Repository Structure

```
.
├── scripts/              # Scripts for running experiments
│   ├── run_inference.py  # Run inference with models
│   ├── run_training.py   # Train distilled models
│   └── load_datasets.py  # Load and prepare datasets
├── src/                  # Main source code
│   ├── data/             # Data processing utilities
│   ├── evaluation/       # Evaluation metrics and procedures
│   ├── models/           # Model loading and interaction
│   ├── prompts/          # Task-specific prompts 
│   └── utils/            # Utility functions
├── init.sh               # Environment initialization script
└── README.md             # This file
```

## Getting Started

### Prerequisites
- Python 3.11+
- CUDA-capable GPU for optimal performance (especially for larger models)
- Git

### Environment Setup

**Clone the repository**
```bash
git clone https://github.com/hendrik-spl/cbs-thesis-efficient-llm-distillation.git
cd cbs-thesis-efficient-llm-distillation
```

**Create environment file**
```bash
cp .env.example .env
# Edit .env to add your API keys (WANDB_API_KEY, HF_TOKEN)
```

**Initialize the environment**
```bash
source init.sh
```
This script handles:
- Installing dependencies via `uv`
- Setting up environment variables
- Installing Ollama (if not already present)
- Starting the Ollama server

**Set up Weights & Biases**
```bash
wandb login
# Enter your API key when prompted
```

## Running Experiments

### Inference
Run inference using a pre-trained model on a specific dataset:

```bash
uv run scripts/run_inference.py --model_name llama3.2:1b --dataset sentiment
```

**Parameters:**
- `--model_name`: Model to use (e.g., llama3.2:1b, llama3.3:70b)
- `--dataset`: Dataset to run inference on (`sentiment`, `gold`, `summary`)
- `--limit`: Number of samples to process
- `--run_on_test`: Whether to run on test set (default: `False`)
- `--use_ollama`: Whether to use Ollama (default: `False`, uses HF)

### Training (Knowledge Distillation)
Train a student model through knowledge distillation:

```bash
uv run scripts/run_training.py --student_model llama3.2:1b --teacher_model llama3.2:1b --dataset sentiment
```

**Or using inference outputs:**
```bash
uv run scripts/run_training.py --student_model llama3.2:1b --teacher_model llama3.3:70b --dataset sentiment:50agree --inference_title noble-sun-21
```

**Parameters:**
- `--student_model`: The model to be distilled (e.g., llama3.2:1b)
- `--teacher_model`: The source model used for distillation (e.g., llama3.3:70b)
- `--dataset`: Dataset to use for distillation
- `--inference_title`: Title of the inference run to use as teacher outputs

## Key Findings

Our experiments demonstrate that:

- Distilled student models consistently outperform equivalently sized raw models, validating knowledge distillation's effectiveness.
- Smaller student models can achieve up to 99% of teacher model performance while reducing energy consumption by up to 99%.
- For complex tasks like summarization, the accuracy gap between teacher and student models widens, but remains acceptable for many applications.
- The initial investment in distillation is typically offset after a few thousand inference queries, with the break-even point achieved more quickly for token-intensive tasks.
- Distilled models offer dramatic improvements in inference speed, making them suitable for latency-sensitive applications.

## Repository Link

Access to repository: [https://github.com/hendrik-spl/sustainable-llm-knowledge-distillation](https://github.com/hendrik-spl/sustainable-llm-knowledge-distillation)
