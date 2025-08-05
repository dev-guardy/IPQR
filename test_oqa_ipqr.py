import torch
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from llms.llama31_8b import get_answers as get_answers_llama31_8b
from llms.llama3_8b import get_answers as get_answers_llama3_8b
from llms.qwen_7b import get_answers as get_answers_qwen_7b
from llms.zephyr_7b import get_answers as get_answers_zephyr_7b
from llms.gemma3_4b import get_answers as get_answers_gemma3_4b
from llms.gemma2_9b import get_answers as get_answers_gemma2_9b
from llms.mistral_7b import get_answers as get_answers_mistral_7b
from llms.chatgpt import get_answers as get_answers_chatgpt
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import os
import torch.nn.functional as F

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

class PowerfulBGEM3(torch.nn.Module):
    def __init__(self, model_name="BAAI/bge-m3"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.projection = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        embeddings = torch.sum(embeddings * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        embeddings = self.projection(self.dropout(embeddings))
        return F.normalize(embeddings, p=2, dim=1)

class BaseBGEM3(torch.nn.Module):
    def __init__(self, model_name="BAAI/bge-m3"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        embeddings = torch.sum(embeddings * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        return F.normalize(embeddings, p=2, dim=1)

def get_args():
    parser = argparse.ArgumentParser(description="Process model configurations for LLM evaluation.")
    parser.add_argument("--name", default='last_oqa_original_0801_onlysft', help="Experiment name")
    parser.add_argument("--add_prompt", default=None, help="Additional prompt to append to each question")
    parser.add_argument("--device_judge", default='cuda:0', help="Device for judging")
    parser.add_argument("--device_rewriter", default='cuda:0', help="Device for rewriting")
    parser.add_argument("--device_completer", default='cuda:0', help="Device for completion")
    parser.add_argument("--device_respond", default='cuda:0', help="Device for responding")
    parser.add_argument("--rewriter_ckpt", default='policy_3840.pt', help="Checkpoint for the rewriter model")
    parser.add_argument("--completer_base_model", default='meta-llama/Llama-3.2-3B-Instruct', help="Base model for completer")
    parser.add_argument("--completer_adapter_dir", default='training_data_llama_improved_0725_v2_5k', help="Adapter directory for completer")
    parser.add_argument("--bgem3_model_path", default='bge-m3-medical-qa-finetuned-hf_0727', help="Path to fine-tuned BGE-M3 model")
    parser.add_argument("--test", default=0, help="Test")
    parser.add_argument("--openai_api_key", default="", help="Openai_api_key")
    return parser.parse_args()

def generate_5_questions(question: str, max_new_tokens: int = 1024) -> list[str]:
    SYSTEM_PROMPT = """You are an expert at improving questions to get better, more complete answers. 
Rewrite the input question into exactly 3 clear improving questions that preserve the original meaning.

Requirements:
- Keep all facts and context identical
- improving questions must maintain original quetsion intent.
- Fix any grammar, spelling, or clarity issues

Output format: Return ONLY valid JSON array, no other text.
["Question 1?", "Question 2?", "Question 3?"]
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this query and generate 3 improved versions: '{question}'"}
    ]
    
    prompt = completer_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = completer_tokenizer(prompt, return_tensors="pt").to(completer.device)

    with torch.no_grad():
        out = completer.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            repetition_penalty=1.1,
            pad_token_id=completer_tokenizer.pad_token_id
        )
    
    gen_text = completer_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    response = gen_text.strip()
    
    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            json_response = response[json_start:json_end]
            questions = json.loads(json_response)
            
            if isinstance(questions, list) and len(questions) >= 3:
                return questions[:3]
            elif isinstance(questions, list) and len(questions) > 0:
                while len(questions) < 3:
                    questions.append(question)
                return questions[:3]
            else:
                return [question] * 3
        else:
            return [question] * 3
            
    except json.JSONDecodeError:
        return [question] * 3

def select_best_question_with_bgem3(original_question: str, generated_questions: list[str], use_finetuned_bgem3: bool = False) -> str:
    try:
        bgem3_model.eval()
        device = next(bgem3_model.parameters()).device
        
        with torch.no_grad():
            original_inputs = bgem3_tokenizer(original_question, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
            original_emb = bgem3_model(original_inputs['input_ids'], original_inputs['attention_mask'])
            
            scores = []
            for i, question in enumerate(generated_questions):
                question_inputs = bgem3_tokenizer(question, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
                question_emb = bgem3_model(question_inputs['input_ids'], question_inputs['attention_mask'])
                score = F.cosine_similarity(original_emb, question_emb, dim=1).item()
                scores.append(score)
            
            best_idx = np.argmax(scores)
            best_question = generated_questions[best_idx]
            return best_question
            
    except Exception:
        return generated_questions[0] if generated_questions else original_question

def complete_question(question: str, max_new_tokens: int = 512, use_finetuned_bgem3: bool = False) -> str:
    generated_questions = generate_5_questions(question, max_new_tokens)
    import random
    best_question = random.choice(generated_questions)
    return best_question

def load_bgem3_model(model_path, device, use_finetuned_bgem3=False):
    if use_finetuned_bgem3:
        model = PowerfulBGEM3("BAAI/bge-m3")
        
        if os.path.exists(model_path):
            model = BaseBGEM3(model_path)
        else:
            model = BaseBGEM3("BAAI/bge-m3")
    else:
        model = BaseBGEM3("BAAI/bge-m3")
    
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return model, tokenizer

def save_questions_to_jsonl(questions, file_path, stage_name):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, question in enumerate(questions):
            data = {
                "id": i,
                "stage": stage_name,
                "question": question
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def ask_rewriter(prompt):   
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    input_ids = rewriter_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(rewriter.device)
    
    terminators = [
        rewriter_tokenizer.eos_token_id,
        rewriter_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = rewriter.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample = False
    )
    response = outputs[0][input_ids.shape[-1]:]
    return rewriter_tokenizer.decode(response, skip_special_tokens=True)

def rewrite_question(question):
    a = ask_rewriter('Rewriting question to make it more understandable, just give me the rewritten question without any other word: ' + question)
    return a

def get_score(question_list, answer_list):       
    reward = []
    reward_device = next(reward_model.parameters()).device
    
    with torch.no_grad():
        for prompt, chosen in tqdm(zip(question_list, answer_list)):
            tokenized_input = reward_tokenizer("prompter: {} assistant: {}".format(prompt, chosen), return_tensors='pt')
            tokenized_input = {k: v.to(reward_device) for k, v in tokenized_input.items()}
            
            logits = reward_model(**tokenized_input).logits
            reward.append(float(torch.sigmoid(logits)[0][0]))
    return np.array(reward).mean()

def save_result(score_list, llm_list_finish, name):
    df = pd.DataFrame(score_list)
    df.index = llm_list_finish
    df.to_csv('test_result/' + name + '.csv')

def check_cuda_devices():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        return device_count
    else:
        return 0

def validate_device(device_str, device_count):
    if device_str.startswith('cuda:'):
        device_num = int(device_str.split(':')[1])
        if device_num >= device_count:
            return 'cuda:0'
    return device_str

def run_evaluation(args, questions_list, is_complete, is_rewrite, use_finetuned_bgem3, run_name):
    add_prompt = args.add_prompt
    device_judge = args.device_judge
    device_rewriter = args.device_rewriter
    device_completer = args.device_completer
    device_respond = args.device_respond
    rewriter_ckpt = args.rewriter_ckpt
    completer_base_model = args.completer_base_model
    completer_adapter_dir = args.completer_adapter_dir
    bgem3_model_path = args.bgem3_model_path
    openai_api_key = args.openai_api_key
    
    device_count = check_cuda_devices()
    device_rewriter = validate_device(device_rewriter, device_count)
    device_completer = validate_device(device_completer, device_count)
    device_respond = validate_device(device_respond, device_count)
    device_judge = validate_device(device_judge, device_count)

    global reward_tokenizer
    global reward_model
    global rewriter
    global rewriter_tokenizer
    global completer
    global completer_tokenizer
    global bgem3_model
    global bgem3_tokenizer

    with torch.cuda.device(device_judge):
        peft_model_id = "vincentmin/llama-2-7b-reward-oasst1"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=1,
            load_in_4bit=True,
            torch_dtype=torch.float16,
        )
        reward_model = PeftModel.from_pretrained(model, peft_model_id)
        reward_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_auth_token=True)
        reward_model.to(device_judge)

    if is_complete:
        bgem3_model, bgem3_tokenizer = load_bgem3_model(bgem3_model_path, device_completer, use_finetuned_bgem3)

    if is_complete:
        if not PEFT_AVAILABLE:
            return False
            
        completer_tokenizer = AutoTokenizer.from_pretrained(completer_base_model)
        if completer_tokenizer.pad_token is None:
            completer_tokenizer.pad_token = completer_tokenizer.eos_token
        completer_tokenizer.padding_side = "left"

        completer_base = AutoModelForCausalLM.from_pretrained(
            completer_base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        completer_base.to(device_completer)

        try:
            completer = PeftModel.from_pretrained(
                completer_base, 
                completer_adapter_dir,
                device_map={"": device_completer},
                is_trainable=False
            )
            completer.eval()
        except Exception:
            completer = completer_base
            completer.eval()

    os.makedirs('intermediate_results', exist_ok=True)
    save_questions_to_jsonl(questions_list, f'intermediate_results/{run_name}_original_questions.jsonl', 'original')

    if is_complete:
        completed_q = []
        for q in tqdm(questions_list, desc="Completing questions"):
            completed_text = complete_question(q, use_finetuned_bgem3=use_finetuned_bgem3)
            completed_q.append(completed_text)
        
        save_questions_to_jsonl(completed_q, f'intermediate_results/{run_name}_completed_questions.jsonl', 'completed')
        
        if 'completer' in globals():
            del completer
            del completer_base
            torch.cuda.empty_cache()
    else:
        completed_q = questions_list[:]

    if is_rewrite:
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        custom_weights_path = rewriter_ckpt
        device = device_rewriter
        
        rewriter_tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        rewriter = AutoModelForCausalLM.from_pretrained(model_id)
        custom_state_dict = torch.load(custom_weights_path, map_location="cpu")
        rewriter.load_state_dict(custom_state_dict['state'])
        rewriter = rewriter.to(dtype=torch.bfloat16)
        rewriter.to(device)

        rewrite_q = []
        for q in tqdm(completed_q, desc="Rewriting"):
            rewrite_q.append(rewrite_question(q))
        
        save_questions_to_jsonl(rewrite_q, f'intermediate_results/{run_name}_rewritten_questions.jsonl', 'rewritten')
    else:
        rewrite_q = completed_q[:]

    llm_list = ['llama31_8b', 'qwen_7b', 'zephyr_7b', 'mistral_7b', 'gemma2_9b', 'gpt35']

    try:
        df = pd.read_csv('test_result/' + run_name + '.csv', index_col=0)
        llm_list_finish = df.index.tolist()
        score_list = df.values.tolist()
    except:
        llm_list_finish = []
        score_list = []

    for llm in llm_list:
        if llm in llm_list_finish:
            continue
            
        with torch.cuda.device(device_respond):
            torch.cuda.empty_cache()

        try:
            if llm == 'llama31_8b':
                if add_prompt:
                    rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_llama31_8b(device_respond, rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_llama31_8b(device_respond, rewrite_q)
                score_list.append([get_score(rewrite_q, rewrite_a)])
                llm_list_finish.append('llama31_8b')
                save_result(score_list, llm_list_finish, run_name)
                
            elif llm == 'qwen_7b':
                if add_prompt:
                    rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_qwen_7b(device_respond, rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_qwen_7b(device_respond, rewrite_q)
                score_list.append([get_score(rewrite_q, rewrite_a)])
                llm_list_finish.append('qwen_7b')
                save_result(score_list, llm_list_finish, run_name)
                
            elif llm == 'zephyr_7b':
                if add_prompt:
                    rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_zephyr_7b(device_respond, rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_zephyr_7b(device_respond, rewrite_q)
                score_list.append([get_score(rewrite_q, rewrite_a)])
                llm_list_finish.append('zephyr_7b')
                save_result(score_list, llm_list_finish, run_name)
                
            elif llm == 'mistral_7b':
                if add_prompt:
                    rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_mistral_7b(device_respond, rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_mistral_7b(device_respond, rewrite_q)
                score_list.append([get_score(rewrite_q, rewrite_a)])
                llm_list_finish.append('mistral_7b')
                save_result(score_list, llm_list_finish, run_name)
                
            elif llm == 'gemma2_9b':
                if add_prompt:
                    rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_gemma2_9b(device_respond, rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_gemma2_9b(device_respond, rewrite_q)
                score_list.append([get_score(rewrite_q, rewrite_a)])
                llm_list_finish.append('gemma2_9b')
                save_result(score_list, llm_list_finish, run_name)
                
            elif llm == 'gpt35':
                if add_prompt:
                    rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-3.5-turbo-1106', rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-3.5-turbo-1106', rewrite_q)
                score_list.append([get_score(rewrite_q, rewrite_a)])
                llm_list_finish.append('gpt35')
                save_result(score_list, llm_list_finish, run_name)
                
        except Exception:
            score_list.append([-100])
            llm_list_finish.append(llm)
        
        save_result(score_list, llm_list_finish, run_name)

    return len(llm_list_finish) == len(llm_list)

def main():
    args = get_args()
    test = args.test

    questions_list = pd.read_csv('datasets/OQA/oasst1_test.csv')['prompt'].tolist()

    if int(test):
        questions_list = [questions_list[i] for i in [0,1]]

    configs = [
        {"is_complete": True, "is_rewrite": False, "use_finetuned_bgem3": True},   
    ]

    for i, config in enumerate(configs, 1):
        is_complete = config["is_complete"]
        is_rewrite = config["is_rewrite"]
        use_finetuned_bgem3 = config["use_finetuned_bgem3"]
        
        config_name = f"comp{int(is_complete)}_rewrite{int(is_rewrite)}_bgefine{int(use_finetuned_bgem3)}"
        run_name = f"{args.name}_{config_name}"
        
        run_completed = run_evaluation(args, questions_list, is_complete, is_rewrite, use_finetuned_bgem3, run_name)
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()