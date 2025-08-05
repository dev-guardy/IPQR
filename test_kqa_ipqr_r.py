import torch
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from tqdm.contrib import tzip
from llms.llama31_8b import get_answers as get_answers_llama31_8b
from llms.llama3_8b import get_answers as get_answers_llama3_8b
from llms.qwen_7b import get_answers as get_answers_qwen_7b
from llms.zephyr_7b import get_answers as get_answers_zephyr_7b
from llms.gemma3_4b import get_answers as get_answers_gemma3_4b
from llms.gemma2_9b import get_answers as get_answers_gemma2_9b
from llms.mistral_7b import get_answers as get_answers_mistral_7b
from llms.chatgpt import get_answers as get_answers_chatgpt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import os
import openai
import time
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

openai_api_key = None
tokenizer = None
bgem3_model = None
bgem3_tokenizer = None
completer = None
completer_tokenizer = None
rewriter = None
judge_model = None
judge_tokenizer = None
global_use_finetuned_bgem3 = False

class BaseBGEM3(torch.nn.Module):
    def __init__(self, model_name="BAAI/bge-m3"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return F.normalize(embeddings, p=2, dim=1)

def get_args():
    parser = argparse.ArgumentParser(description="Process model configurations for LLM evaluation.")
    parser.add_argument("--base_name", default='last_kqa_0802_onlysft', help="Base experiment name")
    parser.add_argument("--use_quantization", type=int, default=0, help="Flag to enable 4-bit quantization for completer")
    parser.add_argument("--add_prompt", default=None, help="Additional prompt to append to each question")
    parser.add_argument("--device_judge", default='cuda:1', help="Device for judge model")
    parser.add_argument("--device_rewriter", default='cuda:1', help="Device for rewriting")
    parser.add_argument("--device_completer", default='cuda:1', help="Device for completion")
    parser.add_argument("--device_respond", default='cuda:1', help="Device for responding")
    parser.add_argument("--rewriter_ckpt", default='policy_3840.pt', help="Checkpoint for the rewriter model")
    parser.add_argument("--completer_base_model", default='meta-llama/Llama-3.2-3B-Instruct', help="Base model for completer")
    parser.add_argument("--completer_adapter_dir", default='training_data_llama_improved_0725_v2_5k', help="Adapter directory for completer")
    parser.add_argument("--bgem3_model_path", default='bge-m3-medical-qa-finetuned-hf_0727', help="Path to fine-tuned BGE-M3 model")
    parser.add_argument("--test", default=0, help="Test mode")
    parser.add_argument("--openai_api_key", default="", help="open_api_key")
    return parser.parse_args()

def ask_judge(prompt):   
    global judge_model, judge_tokenizer
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    input_ids = judge_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(judge_model.device)
    
    terminators = [
        judge_tokenizer.eos_token_id,
        judge_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = judge_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=False
    )
    response = outputs[0][input_ids.shape[-1]:]
    return judge_tokenizer.decode(response, skip_special_tokens=True)

def ask_judge_sample(prompt):   
    global judge_model, judge_tokenizer
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    input_ids = judge_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(judge_model.device)
    
    terminators = [
        judge_tokenizer.eos_token_id,
        judge_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = judge_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return judge_tokenizer.decode(response, skip_special_tokens=True)

def ask_rewriter(prompt):   
    global rewriter, tokenizer
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(rewriter.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = rewriter.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample = False
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def generate_5_questions(question: str, max_new_tokens: int = 1024) -> list[str]:
    global completer, completer_tokenizer
    
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
            temperature=0.6,
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

def select_best_question_with_bgem3(original_question: str, generated_questions: list[str]) -> str:
    global bgem3_model, bgem3_tokenizer, global_use_finetuned_bgem3
    
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

def complete_question(question: str, max_new_tokens: int = 512) -> str:
    generated_questions = generate_5_questions(question, max_new_tokens)
    import random
    best_question = random.choice(generated_questions)
    return best_question

def read_questions_from_jsonl(file_path):
    questions = [] 
    with open(file_path, 'r', encoding='utf-8') as file: 
        for line in file:
            data = json.loads(line) 
            questions.append(data["Question"])
    return questions

def read_mh_from_jsonl(file_path):
    mh_list = []
    with open(file_path, 'r', encoding='utf-8') as file: 
        for line in file: 
            data = json.loads(line) 
            mh_list.append(data['Must_have'])
    return mh_list

def rewrite_question(question):
    a = ask_rewriter('Rewriting question to make it more understandable, just give me the rewritten question without any other word: ' + question)
    return a

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

def most_frequent(entail_result_temp):
    if len(entail_result_temp) == 0:
        return 0
    count = Counter(entail_result_temp)
    max_count = max(count.values())
    frequent_elements = [key for key, value in count.items() if value == max_count]
    if len(frequent_elements) > 1:
        return 1
    return frequent_elements[0]

def is_entails(question, llm_answer, answer):
    with open(f"datasets/K-QA/prompts/is_entails.txt", "r") as f:
        prompt_template_entails = f.read()

    prompt = prompt_template_entails.replace('{question}', question).replace('{llm_answer}', llm_answer).replace('{answer}', answer)
    
    try:
        response = ask_judge(prompt)
        if 'answer is False' in response or 'answer is false' in response or 'the answer as False' in response:
            return 0
        elif 'answer is True' in response or 'answer is true' in response:
            return 1
        else:
            entail_result_temp = []
            for i in range(3):
                try:
                    response = ask_judge_sample(prompt)
                    if 'answer is False' in response or 'answer is false' in response or 'the answer as False' in response:
                        entail_result_temp.append(0)
                    elif 'answer is True' in response or 'answer is true' in response:
                        entail_result_temp.append(1)
                    if len(entail_result_temp) == 3:
                        break
                except Exception:
                    continue
            return most_frequent(entail_result_temp)
    except Exception:
        return 0

def is_contradict(question, llm_answer, answer):
    with open(f"datasets/K-QA/prompts/is_contradict.txt", "r") as f:
        prompt_template_contradict = f.read()

    prompt = prompt_template_contradict.replace('{question}', question).replace('{llm_answer}', llm_answer).replace('{answer}', answer)
    
    try:
        response = ask_judge(prompt)
        if 'answer is False' in response or 'answer is false' in response or 'the answer as False' in response:
            return 0
        elif 'answer is True' in response or 'answer is true' in response:
            return 1
        else:
            entail_result_temp = []
            for i in range(3):
                try:
                    response = ask_judge_sample(prompt)
                    if 'answer is False' in response or 'answer is false' in response or 'the answer as False' in response:
                        entail_result_temp.append(0)
                    elif 'answer is True' in response or 'answer is true' in response:
                        entail_result_temp.append(1)
                    if len(entail_result_temp) == 3:
                        break
                except Exception:
                    continue
            return most_frequent(entail_result_temp)
    except Exception:
        return 0

def get_score(question_list, answer_list, mh_list):
    evaluate_entails = []
    for question, llm_answer, mh in tzip(question_list, answer_list, mh_list):
        result_single = []
        for answer in mh:
            result_single.append(is_entails(question, llm_answer, answer))
        evaluate_entails.append(np.array(result_single).sum() / len(mh))

    evaluate_contradict = []
    for question, llm_answer, mh in tzip(question_list, answer_list, mh_list):
        result_single = []
        for answer in mh:
            result_single.append(is_contradict(question, llm_answer, answer))
        evaluate_contradict.append(np.array(result_single).sum())  

    return np.array(evaluate_entails).mean(), np.array(evaluate_contradict).mean()

def save_result(score_list, llm_list_finish, name):
    columns = ['comp', 'cont'] 
    df = pd.DataFrame(score_list, columns=columns)
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

def load_judge_model(device):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )
    model.to(device)
    return model, tokenizer

def load_bgem3_model(model_path, device, use_finetuned=True):
    if use_finetuned and os.path.exists(model_path):
        try:
            model = BaseBGEM3("BAAI/bge-m3")
            simple_model_path = f"{model_path}"
            model.model = AutoModel.from_pretrained(simple_model_path)
            model.to(device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            return model, tokenizer, True
        except Exception:
            pass
    
    try:
        model = BaseBGEM3("BAAI/bge-m3")
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        return model, tokenizer, False
    except Exception as e:
        raise e

def run_single_config(args, is_complete, is_rewrite, use_finetuned_bgem3):
    global openai_api_key, tokenizer, bgem3_model, bgem3_tokenizer, completer, completer_tokenizer, rewriter
    global judge_model, judge_tokenizer, global_use_finetuned_bgem3
    
    openai_api_key = args.openai_api_key
    global_use_finetuned_bgem3 = use_finetuned_bgem3
    
    config_name = f"comp{int(is_complete)}_rewrite{int(is_rewrite)}_bgefine{int(use_finetuned_bgem3)}"
    experiment_name = f"{args.base_name}_{config_name}"
    
    device_count = check_cuda_devices()
    device_judge = validate_device(args.device_judge, device_count)
    device_rewriter = validate_device(args.device_rewriter, device_count)
    device_completer = validate_device(args.device_completer, device_count)
    device_respond = validate_device(args.device_respond, device_count)
    
    judge_model, judge_tokenizer = load_judge_model(device_judge)
    
    if is_complete:
        bgem3_model, bgem3_tokenizer, loaded_finetuned = load_bgem3_model(
            args.bgem3_model_path, device_completer, use_finetuned_bgem3
        )

    if is_complete:
        if not PEFT_AVAILABLE:
            return
            
        completer_tokenizer = AutoTokenizer.from_pretrained(args.completer_base_model)
        if completer_tokenizer.pad_token is None:
            completer_tokenizer.pad_token = completer_tokenizer.eos_token
        completer_tokenizer.padding_side = "left"

        completer_base = AutoModelForCausalLM.from_pretrained(
            args.completer_base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        completer_base.to(device_completer)

        try:
            completer = PeftModel.from_pretrained(
                completer_base, 
                args.completer_adapter_dir,
                device_map={"": device_completer},
                is_trainable=False
            )
            completer.eval()
        except Exception:
            completer = completer_base
            completer.eval()

    if is_rewrite:
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        custom_weights_path = args.rewriter_ckpt
        device = device_rewriter
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        rewriter = AutoModelForCausalLM.from_pretrained(model_id)
        custom_state_dict = torch.load(custom_weights_path, map_location="cpu")
        rewriter.load_state_dict(custom_state_dict['state'])
        rewriter = rewriter.to(dtype=torch.bfloat16)
        rewriter.to(device)

    np.random.seed(1024)
    indices = np.arange(201)
    test_indices = np.random.choice(indices, size=50, replace=False)
    test_indices = test_indices.tolist()

    file_path = 'datasets/K-QA/dataset/questions_w_answers_broken_4type.jsonl'  
    questions_list = read_questions_from_jsonl(file_path)
    mh_list = read_mh_from_jsonl(file_path)

    mh_list = [mh_list[i] for i in test_indices]
    questions_list = [questions_list[i] for i in test_indices]

    if int(args.test):
        mh_list = [mh_list[i] for i in [0,1]]
        questions_list = [questions_list[i] for i in [0,1]]

    save_questions_to_jsonl(questions_list, f'intermediate_results/{experiment_name}_original_questions.jsonl', 'original')

    if is_complete:
        completed_q = []
        for q in tqdm(questions_list, desc="Completing questions"):
            completed_text = complete_question(q)
            completed_q.append(completed_text)
        
        save_questions_to_jsonl(completed_q, f'intermediate_results/{experiment_name}_completed_questions.jsonl', 'completed')
        
        if 'completer' in globals():
            del completer
            del completer_base
            torch.cuda.empty_cache()
    else:
        completed_q = questions_list[:]

    if is_rewrite:
        rewrite_q = []
        for q in tqdm(completed_q, desc="Rewriting questions"):
            rewrite_q.append(rewrite_question(q))
        
        save_questions_to_jsonl(rewrite_q, f'intermediate_results/{experiment_name}_rewritten_questions.jsonl', 'rewritten')
    else:
        rewrite_q = completed_q[:]

    llm_list = ['llama31_8b','qwen_7b','zephyr_7b','mistral_7b','gemma2_9b','gpt35']
    try:
        df = pd.read_csv('test_result/' + experiment_name + '.csv', index_col=0)
        llm_list_finish = df.index.tolist()
        score_list = df.values.tolist()
    except:
        llm_list_finish = []
        score_list = []

    for llm in llm_list:
        with torch.cuda.device(device_respond):
            torch.cuda.empty_cache()

        if llm == 'llama3_8b' and 'llama3_8b' not in llm_list_finish:
            if args.add_prompt:
                rewrite_q_prompt = [i + args.add_prompt for i in rewrite_q]
                rewrite_a = get_answers_llama3_8b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_llama3_8b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('llama3_8b')
            save_result(score_list, llm_list_finish, experiment_name)

        if llm == 'llama31_8b' and 'llama31_8b' not in llm_list_finish:
            if args.add_prompt:
                rewrite_q_prompt = [i + args.add_prompt for i in rewrite_q]
                rewrite_a = get_answers_llama31_8b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_llama31_8b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('llama31_8b')
            save_result(score_list, llm_list_finish, experiment_name)

        if llm == 'qwen_7b' and 'qwen_7b' not in llm_list_finish:
            if args.add_prompt:
                rewrite_q_prompt = [i + args.add_prompt for i in rewrite_q]
                rewrite_a = get_answers_qwen_7b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_qwen_7b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('qwen_7b')
            save_result(score_list, llm_list_finish, experiment_name)
            
        if llm == 'zephyr_7b' and 'zephyr_7b' not in llm_list_finish:
            if args.add_prompt:
                rewrite_q_prompt = [i + args.add_prompt for i in rewrite_q]
                rewrite_a = get_answers_zephyr_7b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_zephyr_7b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('zephyr_7b')
            save_result(score_list, llm_list_finish, experiment_name)
            
        if llm == 'gemma3_4b' and 'gemma3_4b' not in llm_list_finish:
            if args.add_prompt:
                rewrite_q_prompt = [i + args.add_prompt for i in rewrite_q]
                rewrite_a = get_answers_gemma3_4b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_gemma3_4b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('gemma3_4b')
            save_result(score_list, llm_list_finish, experiment_name)
        
        if llm == 'gemma2_9b' and 'gemma2_9b' not in llm_list_finish:
            if args.add_prompt:
                rewrite_q_prompt = [i + args.add_prompt for i in rewrite_q]
                rewrite_a = get_answers_gemma2_9b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_gemma2_9b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('gemma2_9b')
            save_result(score_list, llm_list_finish, experiment_name)

        if llm == 'mistral_7b' and 'mistral_7b' not in llm_list_finish:
            if args.add_prompt:
                rewrite_q_prompt = [i + args.add_prompt for i in rewrite_q]
                rewrite_a = get_answers_mistral_7b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_mistral_7b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('mistral_7b')
            save_result(score_list, llm_list_finish, experiment_name)

        if llm == 'gpt35' and 'gpt35' not in llm_list_finish:
            try:
                if args.add_prompt:
                    rewrite_q_prompt = [i + args.add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-3.5-turbo-1106', rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-3.5-turbo-1106', rewrite_q)
                score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            except Exception:
                score_list.append([-100, -100])
            llm_list_finish.append('gpt35')
            save_result(score_list, llm_list_finish, experiment_name)
            
        if llm == 'gpt4o' and 'gpt4o' not in llm_list_finish:
            try:
                if args.add_prompt:
                    rewrite_q_prompt = [i + args.add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-4o-2024-05-13', rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-4o-2024-05-13', rewrite_q)
                score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            except Exception:
                score_list.append([-100, -100])
            llm_list_finish.append('gpt4o')
            save_result(score_list, llm_list_finish, experiment_name)
    
    if 'rewriter' in globals():
        del rewriter
    if 'bgem3_model' in globals():
        del bgem3_model
    if 'judge_model' in globals():
        del judge_model
    torch.cuda.empty_cache()
    
    return experiment_name

def main():
    args = get_args()
    
    configs = [
       {"is_complete": True, "is_rewrite": False, "use_finetuned_bgem3": True},   
    ]
    
    all_results = {}
    
    for i, config in enumerate(configs, 1):
        try:
            experiment_name = run_single_config(
                args, 
                config["is_complete"], 
                config["is_rewrite"], 
                config["use_finetuned_bgem3"]
            )
            all_results[experiment_name] = config
            
        except Exception:
            continue
    
    try:
        comparison_data = []
        for experiment_name in all_results.keys():
            try:
                df = pd.read_csv(f'test_result/{experiment_name}.csv', index_col=0)
                config = all_results[experiment_name]
                
                for model in df.index:
                    comparison_data.append({
                        'experiment': experiment_name,
                        'model': model,
                        'comp_score': df.loc[model, 'comp'],
                        'cont_score': df.loc[model, 'cont'],
                        'is_complete': config['is_complete'],
                        'is_rewrite': config['is_rewrite'],
                        'use_finetuned_bgem3': config['use_finetuned_bgem3']
                    })
            except Exception:
                pass
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(f'test_result/{args.base_name}_comparison.csv', index=False)
                
    except Exception:
        pass

if __name__ == "__main__":
    main()