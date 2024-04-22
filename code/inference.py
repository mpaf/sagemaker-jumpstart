
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests


def model_fn(model_dir):
    
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:0")
    
    return model, processor

def predict_fn(data, model_and_processor):
    
    # destruct model and tokenizer
    model, processor = model_and_processor

    # Tokenize sentences
    im_url = data["inputs"]["image"]
    prompt = data["inputs"]["question"]

    image = Image.open(requests.get(im_url, stream=True).raw)
    final_prompt = f"[INST] <image>\n{prompt}[/INST]"

    inputs = processor(final_prompt, image, return_tensors="pt").to("cuda:0")
    
    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    return {"output": processor.decode(output[0], skip_special_tokens=True)}