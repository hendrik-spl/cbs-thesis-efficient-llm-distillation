import torch
from tqdm import tqdm
from transformers import pipeline
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList

from src.models.hf_stopping import KeywordStoppingCriteria
from src.prompts.sentiment import get_sentiment_prompt
from src.models.model_mapping import model_mapping
from src.models.model_utils import find_majority, clean_llm_output_sentiment, query_params_sentiment

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
            sentence = example["sentence"]
            prompt = get_sentiment_prompt(sentence)
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
            print(f"-----------")
            if wandb_run:
                wandb_run.log({
                    "dataset_size": dataset.shape,
                    "sample": i,
                    "prompt": prompt,
                    "student_completion": completion
                    })
            
    @staticmethod
    def query_hf_sc(model, tokenizer, prompt, shots):
        responses = []
        for i in range(shots):
            response = HF_Manager.query_model(model, tokenizer, prompt, query_params_sentiment)
            responses.append(clean_llm_output_sentiment(response))
        
        return find_majority(responses)

    @staticmethod
    def query_model(model, tokenizer, prompt, params):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Tokenize the input prompt and move to the appropriate device
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move inputs to the same device as the models first parameter
        if hasattr(model, "device"):
            device = model.device
        else:
            # For models distributed across multiple devices, get device of first parameter
            param_device = next(model.parameters()).device
            device = param_device
            
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_length = inputs["input_ids"].shape[1]

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
        model_name = model_mapping[model_name]["HF"]

        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     # device_map="auto"
                                                     )
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
    
    @staticmethod
    def load_finetuned_adapter(model_path):
        """
        Load a fine-tuned PEFT/LoRA adapter model from a local path
        """
        import json
        import os
        from peft import PeftModel
        import torch
        
        # Step 1: Load the tokenizer from the fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Step 2: Get the base model name from adapter_config.json
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
        
        print(f"Loading base model: {base_model_name}")
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
        
        # Step 3: Load base model with the EXACT config from the adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Step 4: CRITICAL - Resize token embeddings BEFORE loading adapter
        base_model.resize_token_embeddings(len(tokenizer))
        
        # Step 5: Set pad token ID if it exists
        if tokenizer.pad_token_id is not None:
            base_model.config.pad_token_id = tokenizer.pad_token_id
            print(f"Set pad_token_id to {tokenizer.pad_token_id}")
        
        # Step 6: Now load the adapter
        print(f"Loading adapter from {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        return model, tokenizer