# Resource-Efficient Large Language Model Distillation

Welcome to the repository for our Master’s thesis project, which explores **sustainable and resource-efficient usage of Large Language Models (LLMs)** through **model distillation**. We compare a large teacher LLM against a smaller, distilled student model on real-world tasks to measure both performance and carbon footprint.

---

## Overview

- **Goal**: Investigate how fine-tuned/distilled domain-specific models stack up against large, general-purpose LLMs in terms of:
  - **Accuracy & Task Performance**
  - **Energy Consumption & Carbon Emissions**

- **Initial Approach**:
  1. **Data Preparation** – We gather domain-specific text data, clean it, and tokenize it.
  2. **Teacher Model** – We utilize a large LLM (e.g., Llama 70B) for baseline performance and as a “teacher” for knowledge distillation.
  3. **Student Model** – We fine-tune or distill a smaller pretrained LLM (e.g., Llama 8B) using the teacher’s outputs.
  4. **Evaluation** – We measure performance on test sets and track energy usage/carbon emissions.

  ---

## Getting Started

1. **Initialize the environment**
    ```bash
    source init.sh
    ```
* Note: The `init.sh` script already takes care of setting up and activating the `.venv` environment. The following steps are provided as a fallback.
* This setup is specifically designed for the use of CUDA GPUs. Adaptions might be neccessary depending on available compute resources.

2. **Set up Weights & Biases**
* Run `wandb login` to initialize your login and add your API key when prompted.

3. **Further Instructions will follow as project develops**

## Main Scripts

1. **Run Inference**
    ```python3
    uv run scripts/run_inference.py --model_name llama3.2:1b --dataset sentiment --limit 20
    ```

2. **Run Training**
    ```python3
    uv run scripts/run_training.py --student_model google-t5:t5-small --teacher_model llama3.2:1b --dataset sentiment --json_file_name smooth-surf-5.json
    ```