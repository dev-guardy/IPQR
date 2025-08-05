import pickle
import json
import random
import numpy as np

def generate_answer_pairs(answers, scores, its, s1, s2, s3, s4):
    random.seed(1024)
    
    if len(answers) == 1:
        return []
        
    paired_answers = list(zip(answers, scores, s1, s2))
    
    max_half = [ans for ans in paired_answers if ans[2] >= s3 and ans[3] <= s4 and (ans[2] > s3 or ans[3] < s4)]
    min_half = [ans for ans in paired_answers if ans[2] <= s3 and ans[3] >= s4 and (ans[2] < s3 or ans[3] > s4)]
    
    max_half_sorted = sorted(max_half, key=lambda x: (-x[3], x[2]))
    
    selected_max_half = max_half_sorted[:10]
    
    selected_min_half = random.sample(min_half, min(20, len(min_half)))

    result_pairs = [(a[0], b[0]) for a in selected_max_half for b in selected_min_half]

    return result_pairs

def read_questions_from_jsonl(file_path):
    questions_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            questions_data.append(data)
    return questions_data

def read_broken_questions_from_jsonl(file_path):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            questions.append(data["Question"])
    return questions

with open('data/rewrite/rewrite_kqa_100_train.pkl', 'rb') as f:
    train_rewrite_q = pickle.load(f)

with open('data/rewrite/rewrite_kqa_100_val.pkl', 'rb') as f:
    val_rewrite_q = pickle.load(f)

with open('data/rewrite/rewrite_kqa_100_score_train.pkl', 'rb') as f:
    train_rewrite_s = pickle.load(f)

with open('data/rewrite/rewrite_kqa_100_score_val.pkl', 'rb') as f:
    val_rewrite_s = pickle.load(f)

with open('data/original/entails_kqa_original.pkl', 'rb') as f:
    initial_is_entails = pickle.load(f)

with open('data/original/contradict_kqa_original.pkl', 'rb') as f:
    initial_is_contradict = pickle.load(f)

train_rewrite_ies = [i[0] for i in train_rewrite_s]
train_rewrite_ics = [i[1] for i in train_rewrite_s]
val_rewrite_ies = [i[0] for i in val_rewrite_s]
val_rewrite_ics = [i[1] for i in val_rewrite_s]

try:
    train_rewrite_s_new = np.array(train_rewrite_ies) - 0.5 * np.array(train_rewrite_ics)
except ValueError:
    train_rewrite_s_new = []
    for ie, ic in zip(train_rewrite_ies, train_rewrite_ics):
        if isinstance(ie, (list, np.ndarray)) and isinstance(ic, (list, np.ndarray)):
            s = np.array(ie) - 0.5 * np.array(ic)
        else:
            s = ie - 0.5 * ic
        train_rewrite_s_new.append(s)

try:
    val_rewrite_s_new = np.array(val_rewrite_ies) - 0.5 * np.array(val_rewrite_ics)
except ValueError:
    val_rewrite_s_new = []
    for ie, ic in zip(val_rewrite_ies, val_rewrite_ics):
        if isinstance(ie, (list, np.ndarray)) and isinstance(ic, (list, np.ndarray)):
            s = np.array(ie) - 0.5 * np.array(ic)
        else:
            s = ie - 0.5 * ic
        val_rewrite_s_new.append(s)

initial_s = np.array(initial_is_entails) - np.array(initial_is_contradict)

file_path = 'datasets/K-QA/dataset/questions_w_answers.jsonl'
questions_data = read_questions_from_jsonl(file_path)
question_list = [item["Question"] for item in questions_data]

broken_file_path = 'datasets/K-QA/dataset/questions_w_answers_broken_4type.jsonl'
broken_question_list = read_broken_questions_from_jsonl(broken_file_path)

np.random.seed(1024)
indices = np.arange(len(question_list))
test_indices = np.random.choice(indices, size=50, replace=False)
train_indices = np.setdiff1d(indices, test_indices)
val_indices = np.random.choice(train_indices, size=50, replace=False)
train_indices = np.setdiff1d(train_indices, val_indices)

test_indices = test_indices.tolist()
train_indices = train_indices.tolist()
val_indices = val_indices.tolist()

dpo_data = []

for idx, (i, j, its, s1, s2, s3, s4) in enumerate(zip(train_rewrite_q, train_rewrite_s_new, 
                                                      [initial_s[i] for i in train_indices], 
                                                      train_rewrite_ies, train_rewrite_ics, 
                                                      [initial_is_entails[i] for i in train_indices], 
                                                      [initial_is_contradict[i] for i in train_indices])):
    pairs = generate_answer_pairs(i, j, its, s1, s2, s3, s4)
    if len(pairs) != 0:
        question_idx = train_indices[idx]
        dpo_data.append((question_idx, "train"))

for idx, (i, j, its, s1, s2, s3, s4) in enumerate(zip(val_rewrite_q, val_rewrite_s_new,
                                                      [initial_s[i] for i in val_indices],
                                                      val_rewrite_ies, val_rewrite_ics,
                                                      [initial_is_entails[i] for i in val_indices],
                                                      [initial_is_contradict[i] for i in val_indices])):
    pairs = generate_answer_pairs(i, j, its, s1, s2, s3, s4)
    if len(pairs) != 0:
        question_idx = val_indices[idx]
        dpo_data.append((question_idx, "val"))

unique_dpo_data = list(set(dpo_data))
unique_dpo_data.sort()

print(f"Total unique questions used for DPO: {len(unique_dpo_data)}")
print(f"From train set: {sum(1 for _, split in unique_dpo_data if split == 'train')}")
print(f"From val set: {sum(1 for _, split in unique_dpo_data if split == 'val')}")

with open('data/pairs.jsonl', 'w', encoding='utf-8') as f:
    for idx, split in unique_dpo_data:
        question_data = questions_data[idx]
        freeform_answer = " ".join(question_data.get("Free_form_answer", []))
        must_have = " ".join(question_data.get("Must_have", []))
        nice_to_have = " ".join(question_data.get("Nice_to_have", []))
        
        pair = {
            "question": question_data["Question"],
            "broken": broken_question_list[idx],
            "free_form_answer" : freeform_answer,
            "must_have": must_have,
            "nice_to_have": nice_to_have,
            "split": split
        }
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')

print(f"Generated pairs.jsonl file: data/pairs.jsonl")