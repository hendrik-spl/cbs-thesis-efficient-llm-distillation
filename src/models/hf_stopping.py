from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, keywords, prompt_length):
        self.tokenizer = tokenizer
        self.keywords = keywords
        self.prompt_length = prompt_length
        # Pre-encode each keyword
        self.keyword_ids = []
        for keyword in keywords:
            ids = tokenizer.encode(keyword, add_special_tokens=False)
            if len(ids) > 0:  # Some tokenizers might split a word into multiple tokens
                self.keyword_ids.append(ids)
        
    def __call__(self, input_ids, scores, **kwargs):
        # Only look at generated tokens (ignore prompt)
        generated_ids = input_ids[0][self.prompt_length:]
        
        # Stop if we've generated more than 1-3 tokens (just enough for a sentiment label)
        if len(generated_ids) > 3:
            return True
            
        # Convert to string to check for keywords
        generated_text = self.tokenizer.decode(generated_ids)
        
        # Stop if we see any stopping pattern
        for keyword in self.keywords:
            if keyword in generated_text.lower():
                return True
                
        return False