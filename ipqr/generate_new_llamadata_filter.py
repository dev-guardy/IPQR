import json
import os
from typing import List, Dict, Tuple
from itertools import combinations
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_bge_model():
    print("Loading BGE-M3 model...")
    model = SentenceTransformer('BAAI/bge-m3')
    print("BGE-M3 model loaded successfully!")
    return model

def calculate_bge_similarity(model: SentenceTransformer, text1: str, text2: str) -> float:
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)

def load_improved_data_from_jsonl(jsonl_file: str) -> List[Dict]:
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def extract_improved_questions(item: Dict) -> List[str]:
    improved_questions = []
    
    if 'rewritten_question' in item and item['rewritten_question']:
        improved_questions.append(item['rewritten_question'])
    
    if 'all_improvements' in item and item['all_improvements']:
        for improvement in item['all_improvements']:
            if 'question' in improvement and improvement['question']:
                improved_questions.append(improvement['question'])
    
    improved_questions = list(dict.fromkeys(improved_questions))
    
    return improved_questions

def filter_improved_questions_by_bge(
    model: SentenceTransformer,
    original_question: str, 
    improved_questions: List[str],
    similarity_threshold: float = 0.8
) -> Tuple[List[str], Dict]:
    
    if not improved_questions:
        return [], {"total": 0, "filtered_out": 0, "kept": 0, "similarities": []}
    
    filtered_questions = []
    similarities = []
    
    for improved_q in improved_questions:
        similarity = calculate_bge_similarity(model, original_question, improved_q)
        similarities.append(similarity)
        
        if similarity >= similarity_threshold:
            filtered_questions.append(improved_q)
    
    filter_stats = {
        "total": len(improved_questions),
        "kept": len(filtered_questions),
        "filtered_out": len(improved_questions) - len(filtered_questions),
        "similarities": similarities,
        "avg_similarity": np.mean(similarities) if similarities else 0,
        "min_similarity": np.min(similarities) if similarities else 0,
        "max_similarity": np.max(similarities) if similarities else 0
    }
    
    return filtered_questions, filter_stats

def generate_question_combinations(input_question: str, improved_questions: List[str], is_broken: bool = False, original_question: str = None) -> List[List[str]]:
    
    combinations_list = []
    
    if is_broken and original_question:
        available_questions = improved_questions.copy()
        if original_question not in available_questions:
            available_questions.append(original_question)
    else:
        available_questions = improved_questions.copy()
    
    if len(available_questions) < 3:
        needed = 3 - len(available_questions)
        final_questions = available_questions + [input_question] * needed
        combinations_list.append(final_questions)
    elif len(available_questions) == 3:
        combinations_list.append(available_questions)
    else:
        for combo in combinations(available_questions, 3):
            combinations_list.append(list(combo))
    
    return combinations_list

def create_llama32_training_data_with_bge_filtering(
    input_data: List[Dict], 
    similarity_threshold: float = 0.9,
    output_file: str = "training_data_llama_bge_filtered.jsonl"
):
    
    bge_model = load_bge_model()
    
    SYSTEM_MESSAGE = """You are an expert at improving questions to get better, more complete answers. 
Rewrite the input question into exactly 3 clear improving questions that preserve the original meaning.

Requirements:
- Keep all facts and context identical
- improving questions must maintain original quetsion intent.
- Fix any grammar, spelling, or clarity issues

Output format: Return ONLY valid JSON array, no other text.
["Question 1?", "Question 2?", "Question 3?"]
"""
    
    training_examples = []
    total_filter_stats = {
        "total_items": 0,
        "items_with_improvements": 0,
        "items_after_filtering": 0,
        "total_original_improved": 0,
        "total_filtered_improved": 0,
        "avg_similarity": []
    }
    
    for item_idx, item in enumerate(input_data):
        original_question = item["question"]
        broken_question = item.get("broken", "")
        improved_questions = extract_improved_questions(item)
        
        total_filter_stats["total_items"] += 1
        if improved_questions:
            total_filter_stats["items_with_improvements"] += 1
            total_filter_stats["total_original_improved"] += len(improved_questions)
        
        filtered_improved_questions, filter_stats = filter_improved_questions_by_bge(
            bge_model, original_question, improved_questions, similarity_threshold
        )
        
        if filtered_improved_questions:
            total_filter_stats["items_after_filtering"] += 1
            total_filter_stats["total_filtered_improved"] += len(filtered_improved_questions)
            total_filter_stats["avg_similarity"].extend(filter_stats["similarities"])
        
        if not filtered_improved_questions:
            continue
        
        original_combinations = generate_question_combinations(
            original_question, 
            filtered_improved_questions, 
            is_broken=False
        )
        
        for i, combo in enumerate(original_combinations):
            shuffled_combo = combo.copy()
            random.shuffle(shuffled_combo)
            
            formatted_questions = json.dumps(shuffled_combo, ensure_ascii=False)
            
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE
                },
                {
                    "role": "user", 
                    "content": f"Analyze this query and generate 3 improved versions: '{original_question}'"
                },
                {
                    "role": "assistant",
                    "content": formatted_questions
                }
            ]
            
            training_examples.append({
                "messages": messages,
                "metadata": {
                    "input_type": "original",
                    "original_question": original_question,
                    "combination_id": i + 1,
                    "total_combinations": len(original_combinations),
                    "improved_count": len(filtered_improved_questions),
                    "filtered_from": len(improved_questions),
                    "filter_stats": filter_stats
                }
            })
        
        if broken_question and broken_question.strip():
            broken_combinations = generate_question_combinations(
                broken_question, 
                filtered_improved_questions,
                is_broken=True, 
                original_question=original_question
            )
            
            for i, combo in enumerate(broken_combinations):
                shuffled_combo = combo.copy()
                random.shuffle(shuffled_combo)
                
                formatted_questions = json.dumps(shuffled_combo, ensure_ascii=False)
                
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE
                    },
                    {
                        "role": "user", 
                        "content": f"Analyze this query and generate 3 improved versions: '{broken_question}'"
                    },
                    {
                        "role": "assistant",
                        "content": formatted_questions
                    }
                ]
                
                training_examples.append({
                    "messages": messages,
                    "metadata": {
                        "input_type": "broken",
                        "original_question": original_question,
                        "broken_question": broken_question,
                        "combination_id": i + 1,
                        "total_combinations": len(broken_combinations),
                        "improved_count": len(filtered_improved_questions),
                        "filtered_from": len(improved_questions),
                        "includes_original": True,
                        "filter_stats": filter_stats
                    }
                })
    
    print(f"Total filtering stats:")
    print(f"- Total items: {total_filter_stats['total_items']}")
    print(f"- Items with improvements: {total_filter_stats['items_with_improvements']}")
    print(f"- Items after filtering: {total_filter_stats['items_after_filtering']}")
    print(f"- Original improved questions: {total_filter_stats['total_original_improved']}")
    print(f"- Filtered improved questions: {total_filter_stats['total_filtered_improved']}")
    if total_filter_stats['avg_similarity']:
        print(f"- Average similarity: {np.mean(total_filter_stats['avg_similarity']):.3f}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(training_examples)} training examples to {output_file}")
    return output_file

def create_sample_analysis_with_filtering(input_data: List[Dict], similarity_threshold: float = 0.9) -> Dict:
    bge_model = load_bge_model()
    
    analysis = {
        "total_items": len(input_data),
        "items_with_improvements": 0,
        "items_with_broken": 0,
        "items_after_filtering": 0,
        "improvement_count_distribution": {},
        "filtered_improvement_count_distribution": {},
        "total_combinations_original": 0,
        "total_combinations_broken": 0,
        "total_combinations": 0,
        "filter_efficiency": 0.0
    }
    
    total_original_improved = 0
    total_filtered_improved = 0
    
    for item in input_data:
        original_question = item["question"]
        broken_question = item.get("broken", "")
        improved_questions = extract_improved_questions(item)
        improvement_count = len(improved_questions)
        
        if improvement_count > 0:
            analysis["items_with_improvements"] += 1
            total_original_improved += improvement_count
        
        if broken_question and broken_question.strip():
            analysis["items_with_broken"] += 1
        
        if improvement_count not in analysis["improvement_count_distribution"]:
            analysis["improvement_count_distribution"][improvement_count] = 0
        analysis["improvement_count_distribution"][improvement_count] += 1
        
        if improved_questions:
            filtered_improved_questions, _ = filter_improved_questions_by_bge(
                bge_model, original_question, improved_questions, similarity_threshold
            )
            filtered_count = len(filtered_improved_questions)
            
            if filtered_count > 0:
                analysis["items_after_filtering"] += 1
                total_filtered_improved += filtered_count
            
            if filtered_count not in analysis["filtered_improvement_count_distribution"]:
                analysis["filtered_improvement_count_distribution"][filtered_count] = 0
            analysis["filtered_improvement_count_distribution"][filtered_count] += 1
        else:
            filtered_count = 0
            if 0 not in analysis["filtered_improvement_count_distribution"]:
                analysis["filtered_improvement_count_distribution"][0] = 0
            analysis["filtered_improvement_count_distribution"][0] += 1
    
    if total_original_improved > 0:
        analysis["filter_efficiency"] = (total_original_improved - total_filtered_improved) / total_original_improved
    
    return analysis

def verify_llama32_format_with_filtering(jsonl_file: str, num_samples: int = 3):
    print(f"Verifying {jsonl_file}...")
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total training examples: {len(lines)}")
    
    original_count = 0
    broken_count = 0
    
    for line in lines:
        data = json.loads(line.strip())
        metadata = data.get("metadata", {})
        
        if metadata.get("input_type") == "original":
            original_count += 1
        elif metadata.get("input_type") == "broken":
            broken_count += 1
    
    print(f"- Original questions: {original_count}")
    print(f"- Broken questions: {broken_count}")
    
    print(f"\nFirst {num_samples} examples:")
    for i, line in enumerate(lines[:num_samples]):
        data = json.loads(line.strip())
        messages = data["messages"]
        metadata = data.get("metadata", {})
        
        print(f"\n--- Example {i+1} ({metadata.get('input_type', 'unknown')}) ---")
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                try:
                    questions = json.loads(content)
                    print(f"{role.upper()}:")
                    for j, q in enumerate(questions, 1):
                        print(f"  {j}. {q}")
                except:
                    print(f"{role.upper()}: {content}")
            else:
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"{role.upper()}: {content_preview}")

if __name__ == "__main__":
    print("Starting BGE-M3 filtered Llama 3.2 training data generation...")
    
    input_file = 'improved_pairs.jsonl'
    similarity_threshold = 0.8
    
    print(f"Loading data from {input_file}...")
    
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error at line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        exit(1)
    
    print(f"Loaded {len(data)} items.")
    
    print(f"Analyzing data with BGE filtering (threshold: {similarity_threshold})...")
    analysis = create_sample_analysis_with_filtering(data, similarity_threshold)
    print(f"Analysis results:")
    print(f"- Total items: {analysis['total_items']}")
    print(f"- Items with improvements: {analysis['items_with_improvements']}")
    print(f"- Items after filtering: {analysis['items_after_filtering']}")
    print(f"- Items with broken questions: {analysis['items_with_broken']}")
    print(f"- Filtering efficiency: {analysis['filter_efficiency']*100:.1f}% removed")
    
    print(f"Generating training data with BGE filtering...")
    output_file = create_llama32_training_data_with_bge_filtering(
        data, 
        similarity_threshold=similarity_threshold
    )
    
    verify_llama32_format_with_filtering(output_file)
    
    print(f"Completed! Generated {output_file}")