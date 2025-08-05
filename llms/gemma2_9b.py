import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def ask(model, tokenizer, question):
    """
    Ask a single question to the model and return the response.
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer for the model
        question (str): The question to ask
        
    Returns:
        str: The model's response
    """
    chat = [
        {
            "role": "user", 
            "content": question
        }
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize and move to device
    inputs = tokenizer.encode(
        prompt, 
        add_special_tokens=False, 
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        do_sample=False,
        use_cache=False
    )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0])
    answer = response.split(prompt)[1].replace('<eos>', '')
    
    return answer

def get_answers(device, questions):
    """
    Get answers for multiple questions using Gemma-2-9b-it model.
    
    Args:
        device (str): Device to run model on (e.g., 'cuda', 'cpu', 'auto')
        questions (list): List of questions to ask
        
    Returns:
        list: List of answers corresponding to the questions
    """
    model_id = "google/gemma-2-9b-it"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Get answers for all questions
    answers = []
    for question in questions:
        answer = ask(model, tokenizer, question)
        answers.append(answer)
    
    # Clean up memory
    del model
    torch.cuda.empty_cache()  # Clear CUDA cache
    
    return answers