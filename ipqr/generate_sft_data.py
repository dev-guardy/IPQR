import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import time
from typing import Dict, List, Tuple, Set
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import gc
import hashlib

class QuestionRewriter:
    def __init__(self, 
                 llama_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 embedding_model_name: str = "BAAI/bge-m3"):
        
        print("Loading models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_model.to(self.device)
        
        print("Models loaded successfully!")
    
    def generate_llama_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        try:
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            inputs = self.llama_tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.llama_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.llama_tokenizer.eos_token_id,
                    eos_token_id=self.llama_tokenizer.eos_token_id
                )
            
            response = self.llama_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def get_embedding(self, text: str) -> np.ndarray:
        try:
            inputs = self.embedding_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu().numpy()
            
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return embeddings
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.array([])
    
    def get_question_hash(self, question: str) -> str:
        return hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    def rewrite_question(self, original_question: str, must_have: str, nice_to_have: str, attempt: int = 1) -> str:
        
        import random

        temp = random.uniform(0.5, 1.3)
        

        rewrite_prompt =  f"""You are an expert at improving comprehensive questions to get more comprehensive answers while preserving their original intent, style, and focus.

Original question: "{original_question}"

First, analyze the intent of this original question. The improved comprehensive question must have the same intent as the original question.

Context for answers (DO NOT directly reference these in your rewrite):
- free form answers : {must_have}
- Additional helpful context: {nice_to_have}

Your task: Improve the original question

CRITICAL PRESERVATION RULES:
1. "Keep the EXACT same topic, subject matter, and question focus"
2. "Preserve all key terms and concepts from the original question"
3. DO NOT add new topics, redirect focus, or change the core question type
4. DO NOT explicitly mention or reference the context information above

IMPROVEMENT STRATEGIES (choose what fits best):
- Fix grammar/clarity issues gently 
- Add minimal context ONLY if the original is genuinely unclear
- Make implicit questions more explicit
- Improve sentence structure without changing meaning
- Ensure the question is answerable as intended
- **Favor concise, comprehensive questions - shorter questions that invite broader responses are often better questions**

QUALITY CHECKS:
- **Would someone reading original question and improved question that they think about this questions are same meaning?**
- **Is the improvement minimal and focused on clarity/answerability?**
- Is the question concise yet comprehensive enough to invite thorough responses?

IMPORTANT NOTES:
- "Avoid overfitting to the provided better answers context - the improved question should stand alone and invite broad, thorough responses"
- "Craft a comprehensive question that would naturally prompt answers covering the context"
- Remember: concise, comprehensive questions often generate the best responses

Example 1:
- Original: "I don't know much about solar panels, would you explain them to me?"
- Improved question ex 1:  "What can you tell me about solar panels, including how they work, their costs and benefits?"
- Improved question ex 2: "How do solar panels work?"
Remember original question is "{original_question}". Don't leave the question intent. You just rewritting question.

Example 2:
Original: "A Java method write that takes ArrayList of Student objects and returns HashMap with student ID as key and GPA as value"
Improved question ex 1: "Can you write a Java method that takes an ArrayList of Student objects and returns a HashMap with student ID as key and GPA as value, including null checking?"
Improved question ex 2: "How do I create a Java method to convert a Student ArrayList into a HashMap mapping student IDs to GPAs?"

CRITICAL: Respond with ONLY the Improved question. Do not include any introductory phrases like "Here is a rewritten question", "Here is", or any explanatory text. Just provide the improved question directly.

Improved question:"""

        rewritten = self.generate_llama_response(rewrite_prompt, temperature=temp)
        
        rewritten = rewritten.replace("Rewritten question:", "").strip()
        rewritten = rewritten.split("\n")[0].strip()
        
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
        
        if original_question.endswith("?") and not rewritten.endswith("?"):
            rewritten += "?"
        print(rewritten)    
        return rewritten
    
    def answer_question(self, question: str) -> str:
        
        answer_prompt = question

        answer = self.generate_llama_response(answer_prompt, max_length=400)
        return answer
    
    def calculate_similarity_score(self, answer: str, target_info: str) -> float:
        if not answer.strip() or not target_info.strip():
            return 0.0
        
        try:
            answer_emb = self.get_embedding(answer)
            target_emb = self.get_embedding(target_info)
            
            if answer_emb.size == 0 or target_emb.size == 0:
                return 0.0
            
            similarity = cosine_similarity(answer_emb, target_emb)[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def evaluate_answer(self, answer: str, must_have: str, nice_to_have: str) -> float:
        
        target_info = must_have + " " + nice_to_have
        
        similarity_score = self.calculate_similarity_score(answer, target_info)
        
        must_have_keywords = set(re.findall(r'\b\w+\b', must_have.lower()))
        nice_to_have_keywords = set(re.findall(r'\b\w+\b', nice_to_have.lower()))
        answer_keywords = set(re.findall(r'\b\w+\b', answer.lower()))
        
        must_have_overlap = len(must_have_keywords.intersection(answer_keywords)) / max(len(must_have_keywords), 1)
        nice_to_have_overlap = len(nice_to_have_keywords.intersection(answer_keywords)) / max(len(nice_to_have_keywords), 1)
        
        final_score = similarity_score
        
        return final_score
    
    def process_single_item(self, item: Dict) -> Dict:
        
        original_question = item["question"]
        must_have = item["must_have"]
        nice_to_have = item["nice_to_have"]
        
        print(f"\nProcessing: {original_question}")
        
        original_answer = self.answer_question(original_question)
        original_score = self.evaluate_answer(original_answer, must_have, nice_to_have) * 0.99
        
        print(f"Original score (s1): {original_score:.4f}")
        
        successful_rewrites = []
        seen_hashes = set()
        seen_hashes.add(self.get_question_hash(original_question))
        
        attempt = 0
        max_attempts = 15
        
        while len(successful_rewrites) < 10 and attempt < max_attempts:
            attempt += 1
            print(f"  Attempt {attempt}...")
            
            rewritten_question = self.rewrite_question(original_question, must_have, nice_to_have, attempt)
            question_hash = self.get_question_hash(rewritten_question)
            
            if question_hash in seen_hashes:
                print(f"    Duplicate question, skipping...")
                continue
            
            seen_hashes.add(question_hash)
            
            rewritten_answer = self.answer_question(rewritten_question)
            rewritten_score = self.evaluate_answer(rewritten_answer, must_have, nice_to_have)
            
            print(f"    Rewritten score (sr): {rewritten_score:.4f}")
            print(f"    Question: {rewritten_question}")
            
            if rewritten_score > original_score:
                improvement = rewritten_score - original_score
                successful_rewrites.append({
                    'question': rewritten_question,
                    'score': rewritten_score,
                    'improvement': improvement,
                    'attempt': attempt
                })
                print(f"    Success #{len(successful_rewrites)}: improvement = {improvement:.4f}")
            else:
                print(f"    No improvement: {rewritten_score:.4f} <= {original_score:.4f}")
            
            time.sleep(0.5)
        
        result_item = item.copy()
        result_item["original_score"] = original_score
        
        if successful_rewrites:
            best_rewrite = max(successful_rewrites, key=lambda x: x['improvement'])
            
            result_item["rewritten_question"] = best_rewrite['question']
            result_item["rewritten_score"] = best_rewrite['score']
            result_item["improvement"] = best_rewrite['improvement']
            result_item["successful_attempts"] = len(successful_rewrites)
            result_item["total_attempts"] = attempt
            result_item["all_improvements"] = successful_rewrites
            
            print(f"Final result: {len(successful_rewrites)} successful rewrites found")
            print(f"  Best improvement: {best_rewrite['improvement']:.4f}")
            print(f"  Best question: {best_rewrite['question']}")
        else:
            result_item["rewritten_question"] = original_question
            result_item["rewritten_score"] = original_score
            result_item["improvement"] = 0.0
            result_item["successful_attempts"] = 0
            result_item["total_attempts"] = attempt
            result_item["all_improvements"] = []
            print(f"No improvements found after {attempt} attempts")
        
        return result_item
    
    def process_jsonl_file(self, input_file: str, output_file: str):
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        print(f"Loaded {len(data)} items from {input_file}")
        
        processed_count = 0
        improved_count = 0
        total_improvements = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(data, desc="Processing questions"):
                try:
                    result_item = self.process_single_item(item)
                    
                    f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                    f.flush()
                    
                    processed_count += 1
                    if result_item["improvement"] > 0:
                        improved_count += 1
                        total_improvements += result_item["successful_attempts"]
                    
                    print(f"Progress: {processed_count}/{len(data)}, Improved: {improved_count}, Total improvements: {total_improvements}")
                    
                    if processed_count % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing item: {e}")
                    original_item = item.copy()
                    original_item["error"] = str(e)
                    f.write(json.dumps(original_item, ensure_ascii=False) + '\n')
                    f.flush()
                    continue
        
        print(f"\nCompleted!")
        print(f"Processed: {processed_count}")
        print(f"Successfully improved: {improved_count}")
        print(f"Total successful improvements found: {total_improvements}")
        print(f"Average improvements per successful item: {total_improvements/max(improved_count, 1):.2f}")
        print(f"Results saved to: {output_file}")

def main():
    rewriter = QuestionRewriter()
    
    input_file = "pairs.jsonl"
    output_file = "improved_pairs.jsonl"
    
    rewriter.process_jsonl_file(input_file, output_file)

if __name__ == "__main__":
    main()