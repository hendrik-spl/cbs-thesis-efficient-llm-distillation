def get_sft_config(student_model, dataset_name, wandb_run, model_output_dir):
    training_args = {
    # Constant parameters
    ## Basics
    "output_dir": model_output_dir,
    "run_name": f"{student_model}_{dataset_name}_{wandb_run.name}",
    "report_to": 'wandb',

    ## Evaluation and Saving
    "eval_strategy": "epoch",
    "logging_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,

    ## Performance
    "seed": 42,
    "packing": False,
    "gradient_checkpointing": True,

    ## Training parameters
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,

    ## Regularization
    "weight_decay": 0.1,
    "max_grad_norm": 0.3,
    }

    # Combine all parameters
    training_args.update(get_model_specific_params(student_model))
    training_args.update(get_dataset_specific_params(dataset_name))

    if "summary" in dataset_name and "70b" not in student_model: # reduce batch size by 2 for this task to fit in memory
        training_args["per_device_train_batch_size"] = training_args["per_device_train_batch_size"] // 2
        training_args["per_device_eval_batch_size"] = training_args["per_device_eval_batch_size"] // 2
        training_args["gradient_accumulation_steps"] = training_args["gradient_accumulation_steps"] * 2

    # Convert to SFTConfig
    # training_args = SFTConfig(**training_args)
    return training_args

def get_model_specific_params(model_name):
    params = {}
    if "1b" in model_name:
        params["per_device_train_batch_size"] = 8
        params["per_device_eval_batch_size"] = 8
        params["gradient_accumulation_steps"] = 8
        params["learning_rate"] = 2e-5
    if "8b" in model_name:
        params["per_device_train_batch_size"] = 4
        params["per_device_eval_batch_size"] = 4
        params["gradient_accumulation_steps"] = 16
        params["learning_rate"] = 1e-5
    if "70b" in model_name:
        params["per_device_train_batch_size"] = 1
        params["per_device_eval_batch_size"] = 1
        params["gradient_accumulation_steps"] = 64
        params["learning_rate"] = 5e-6
    return params

def get_dataset_specific_params(dataset_name):
    params = {}
    if "sentiment" in dataset_name:
        params["max_seq_length"] = 392 # max_new_tokens 8 + max_context_length 384
        params["num_train_epochs"] = 3
    if "gold" in dataset_name:
        params["max_seq_length"] = 864 # max_new_tokens 96 + max_context_length 768
        params["num_train_epochs"] = 3
    if "summary" in dataset_name:
        params["max_seq_length"] = 6400 # max_new_tokens 256 + max_context_length 6144
        params["num_train_epochs"] = 3
    return params