import weave

from src.data.data_manager import get_samples
from src.models.hf_utils import HF_Manager
from src.models.ollama_utils import query_ollama_model
from src.models.query_utils import get_query_params

weave.init("model-inference-v2")

def track_samples(model, dataset_name, use_ollama):
    sample_prompts = get_samples(dataset_name)
    query_params = get_query_params(dataset_name)
    func = query_ollama_model if use_ollama else HF_Manager.query_model

    responses = []
    for sample_prompt in sample_prompts:
        track_sample(func, model, sample_prompt, query_params)

    return responses

@weave.op()
def track_sample(func, model, prompt, query_params):
    response = func(model, prompt, query_params)
    return response