#!/usr/bin/env python3

import json, sys, time, re
from openai import OpenAI
from tqdm import tqdm

SYS_PROMPT_WORD_ORDER = """You are a question scrambler.
Given a clean, well-formed question, rewrite it the way real people might type when they're being careless with grammar.
- Actually change the word order - move words to different positions in the sentence
- Drop articles (a, an, the) and auxiliary verbs (can, do, are) where people commonly do
- Put question words in the middle or end instead of the beginning sometimes
- Rearrange phrases so they sound like someone typing without thinking about proper grammar
- Make real word order mistakes that people actually make when typing quickly
- Keep all important nouns and technical terms but move them around naturally
- Focus on realistic grammar errors - how would someone actually mess up the word order?
- Output ONLY the reordered question as a plain string."""

SYS_PROMPT_ABBREVIATIONS = """You are a text shortener.
Given a clean, well-formed question, rewrite it the way people actually text or type casually online.
- Use common texting shortcuts that real people use frequently
- Apply typical abbreviations and number substitutions people actually type
- Replace common words with their standard text shortcuts
- Remove unnecessary words entirely - drop articles, auxiliary verbs, extra words people skip
- Make the whole sentence much shorter, not just individual words
- Cut out filler words and keep only the essential parts
- Think about the most minimal way to ask the same question
- Make it look like authentic casual texting where people save maximum time and effort
- Keep important technical terms but strip everything else down to basics
- Output ONLY the abbreviated question as a plain string."""

SYS_PROMPT_KEYWORDS = """You are a search query converter.
Given a clean, well-formed question, rewrite it the way people actually type into search engines.
- Make it choppy and fragmented like real search queries
- Focus on the essential keywords people would actually search for
- Remove unnecessary words but keep the core searchable terms
- Think about how real people simplify questions when searching
- Make it feel like authentic search behavior, not artificial keyword stuffing
- Preserve all important subject matter and technical terms
- Keep it practical - how would someone really search for this information?
- Output ONLY the search-style query as a plain string."""

SYS_PROMPT_COLLOQUIAL = """You are a casual speaker converter.
Given a clean, well-formed question, rewrite it the way people actually talk in real conversation.
- Vary the length - sometimes make it very short and direct, sometimes longer with filler words
- For short versions: drop unnecessary words, be very casual and brief
- For longer versions: add realistic filler words, hesitations, and conversational markers
- Use natural speech patterns like people actually speak - not consistently formal or informal
- Include authentic casual language that varies based on confidence level and urgency
- Sometimes be direct and quick, sometimes rambling and uncertain
- Preserve all technical terms and key concepts intact
- Think about how real people's speech length changes based on mood and situation
- Output ONLY the conversational question as a plain string."""

PROMPTS = [
    SYS_PROMPT_WORD_ORDER,
    SYS_PROMPT_ABBREVIATIONS, 
    SYS_PROMPT_KEYWORDS,
    SYS_PROMPT_COLLOQUIAL
]

def break_question(question: str, client, prompt_idx: int, model="gpt-4o"):
    system_prompt = PROMPTS[prompt_idx]
    print(question)
    for attempt in range(5):
        try:
            rsp = client.chat.completions.create(
                model=model,
                temperature=0.7,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question.strip()},
                ],
            )
            text = rsp.choices[0].message.content.strip()
            if 5 < len(text) < 200:
                return text
            raise ValueError("Output too short or too long")
        except Exception as e:
            wait = 2 ** attempt
            print(f"[retry {attempt}] {e}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed to break question: {question}")

def extract_question_from_prompt(prompt_text):
    import re
    match = re.search(r'Q:\s*(.+?)\s*\nA:', prompt_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def replace_question_in_prompt(prompt_text, new_question):
    import re
    return re.sub(r'(Q:\s*)(.+?)(\s*\nA:)', r'\1' + new_question + r'\3', prompt_text, flags=re.DOTALL)

def main(src="testset.jsonl", dst="truthfulqa_test_broken.jsonl"):
    api_key = ""
    if not api_key:
        sys.exit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    with open(dst, "w", encoding="utf-8") as out_f, \
         open(src, "r", encoding="utf-8") as in_f:
        
        line_count = 0
        for line in tqdm(in_f):
            item = json.loads(line)
            prompt_text = item.get("prompt", "").strip()
            if not prompt_text:
                continue
            
            orig_q = extract_question_from_prompt(prompt_text)
            if not orig_q:
                print(f"Could not extract question from prompt: {prompt_text[:100]}...", file=sys.stderr)
                continue
            
            prompt_idx = line_count % 4
            
            try:
                broken_q = break_question(orig_q, client, prompt_idx)
                new_prompt = replace_question_in_prompt(prompt_text, broken_q)
                item["prompt"] = new_prompt
                
                prompt_types = ["word_order", "abbreviations", "keywords", "colloquial"]
                item["broken_type"] = prompt_types[prompt_idx]
                
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                line_count += 1
                
            except Exception as e:
                print(f"Skipping question due to error: {e}", file=sys.stderr)
                continue

    print(f"Done. Broken questions saved to {dst}")

if __name__ == "__main__":
    main()