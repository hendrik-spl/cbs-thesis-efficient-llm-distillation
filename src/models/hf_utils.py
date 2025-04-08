import torch
from tqdm import tqdm
from transformers import pipeline
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList

from src.models.hf_stopping import KeywordStoppingCriteria

class HF_Manager:
    
    @staticmethod
    def predict(model_path, dataset, wandb_run=None, limit=5):

        pipe = pipeline(
            model=model_path,
            task="text-generation",
            )

        print(f"Running test inference with model {model_path} on dataset {dataset.shape}. Limit: {limit}.")
        for i, example in tqdm(enumerate(dataset), total=min(limit, len(dataset))):
            if i >= limit:
                break
            prompt = example["prompt"]
            completion = pipe(prompt, max_new_tokens=10)
            completion = completion[0]["generated_text"]
            label_pos = completion.find("Final Label:")
            if label_pos == -1:
                print("Label not found in completion.")
                continue
            completion = completion[label_pos + len("Final Label:"):].strip()
            print(f"Example {i}:")
            print(f"Prompt: {prompt}")
            print(f"Completion by student: {completion}")
            print(f"Completion by teacher: {example['completion']}")
            print(f"-----------")
            if wandb_run:
                wandb_run.log({
                    "dataset_size": dataset.shape,
                    "sample": i,
                    "prompt": prompt,
                    "student_completion": completion
                    })
            
            
    @staticmethod
    def query_model(model_config, prompt, params):
        model = model_config[0]
        tokenizer = model_config[1]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # model.to(device)

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs.input_ids.shape[1]

        # Create stopping criteria - stop on seeing these keywords or patterns
        stop_words = ["text:"]
        stopping_criteria = StoppingCriteriaList([
            KeywordStoppingCriteria(tokenizer, stop_words, prompt_length)
        ])

        # Generate a response
        with torch.no_grad():
            outputs = model.generate(**inputs,
                                    do_sample=params.get("do_sample"),
                                    temperature=params.get("temperature"),
                                    top_p=params.get("top_p"),
                                    top_k=params.get("top_k"),
                                    max_new_tokens=params.get("max_new_tokens"),
                                    pad_token_id=tokenizer.eos_token_id,
                                    stopping_criteria=stopping_criteria,
                                    )

        # Decode ONLY the generated tokens (exclude the input prompt tokens)
        response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        
        return response

    @staticmethod
    def load_model(model_name: str, peft: bool):
        """
        Loads a model and tokenizer from the Hugging Face model hub.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            model (PreTrainedModel): The model loaded from the model hub.
            tokenizer (PreTrainedTokenizer): The tokenizer loaded from the model hub.
            peft_config (Optional[LoraConfig]): The PEFT configuration for the model, if it exists.
        """
        model_hf_mapping = {
            "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
            "llama3.1:405b": "meta-llama/Llama-3.1-405B-Instruct",
            "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
            "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
            "llama3.3:70b": "meta-llama/Llama-3.3-70B-Instruct",
            "smollm2:135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
        }
        if model_name in model_hf_mapping:
            model_name = model_hf_mapping[model_name]
        else:
            raise ValueError(f"Model name {model_name} not recognized. Predefined models are: {model_hf_mapping.keys()}")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            print(f"Tokenizer {tokenizer.name_or_path} does not have a pad token. Setting a unique pad token.")

            # Add a new special token as pad_token
            special_tokens_dict = {'pad_token': '[PAD]'}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added_toks} special tokens to the tokenizer")
            
            # Resize the model's token embeddings to account for the new pad token
            model.resize_token_embeddings(len(tokenizer))

            # Set the pad_token_id to the ID of the new pad token
            model.config.pad_token_id = tokenizer.pad_token_id

            print(f"tokenizer.pad_token: {tokenizer.pad_token}")
            print(f"model.config.pad_token_id: {model.config.pad_token_id}")
            print(f"model.config.eos_token_id: {model.config.eos_token_id}")

        if peft:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                modules_to_save=["lm_head", "embed_tokens"],
            )
            print(f"Loaded model {model_name} with PEFT configuration.")
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        return model, tokenizer