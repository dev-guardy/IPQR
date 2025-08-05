
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

def ask(model, processor, question):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user", 
            "content": [{"type": "text", "text": question}]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=False
        )
    
    generation = generation[0][input_len:]
    response = processor.decode(generation, skip_special_tokens=True)
    
    return response

def get_answers(device, questions):
    model_id = "google/gemma-3-4b-it"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True
    ).eval()
    
    answers = [ask(model, processor, question) for question in questions]
    
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return answers