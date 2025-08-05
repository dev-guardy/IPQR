import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def ask(model, tokenizer, question):
    # 올바른 방법: messages 형태로 구성하고 chat template 사용
    messages = [
        {"role": "user", "content": question}
    ]
    
    # chat template 적용
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    output = model.generate(
        inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False
    )
    
    # 입력 토큰 길이만큼 제외하고 새로 생성된 부분만 디코딩
    new_tokens = output[0][inputs.shape[1]:]
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return decoded_output.strip()

def get_answers(device, questions):
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    model.to(device)
    
    answers = [ask(model, tokenizer, question) for question in questions]
    
    del model
    torch.cuda.empty_cache()
    return answers