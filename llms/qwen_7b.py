import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def ask(model, tokenizer, question):
    # Qwen2.5 chat template 사용
    messages = [
        {"role": "user", "content": question}
    ]
    
    # apply_chat_template으로 올바른 형식 생성
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Qwen2.5에 맞는 generation 설정
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.1
    )
    
    # 입력 토큰 길이만큼 제거하고 디코딩
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return decoded_output.strip()

def get_answers(device, questions):
    # Qwen2.5-7B-Instruct 모델 사용
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer_ = AutoTokenizer.from_pretrained(model_id)
    
    # pad_token 설정 (Qwen2.5는 보통 eos_token과 동일)
    if tokenizer_.pad_token is None:
        tokenizer_.pad_token = tokenizer_.eos_token
    
    model_ = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True  # Qwen 모델에 필요할 수 있음
    )
    
    model_.to(device)

    answers = [ask(model_, tokenizer_, question) for question in questions]

    del model_  # Explicitly delete the model to free up CUDA memory
    torch.cuda.empty_cache()  # Clear cache for Qwen
    return answers