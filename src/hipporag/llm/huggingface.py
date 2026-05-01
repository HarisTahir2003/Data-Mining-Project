import torch
from .base import BaseLLM

class HuggingFaceLLM(BaseLLM):
    """
    Implementation of BaseLLM using local HuggingFace models.
    Default uses Qwen2.5-7B-Instruct with 4-bit quantization to fit on free Colab GPUs.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        except ImportError:
            raise ImportError("Please install transformers, torch, accelerate, and bitsandbytes to use HuggingFaceLLM.")
            
        self.model_name = model_name
        print(f"Loading {model_name} in 4-bit quantization...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=2048,
            do_sample=False, # greedy decoding for information extraction
            return_full_text=False
        )
        
    def generate(self, prompt: str) -> str:
        # Construct chat template for Instruct models
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        prompt_templated = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        outputs = self.pipe(prompt_templated)
        return outputs[0]["generated_text"]
