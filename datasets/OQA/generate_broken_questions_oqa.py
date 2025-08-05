#!/usr/bin/env python3

import csv
import json
import sys
import time
import re
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
    print(f"Processing: {question[:50]}...")
    
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
            if 5 < len(text) < 500:
                return text
            raise ValueError(f"Output length issue: {len(text)} chars")
        except Exception as e:
            wait = 2 ** attempt
            print(f"[retry {attempt}] {e}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed to break question: {question[:50]}...")

def main(src="oasst1_test.csv", dst="oasst1_broken_test.csv"):
    api_key = ""
    
    if not api_key or api_key == "your-openai-api-key-here":
        sys.exit("Please set your OpenAI API key in the script")
    
    client = OpenAI(api_key=api_key)
    
    try:
        with open(src, 'r', encoding='utf-8', newline='') as infile:
            sample = infile.read(1024)
            infile.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(infile, delimiter=delimiter)
            fieldnames = reader.fieldnames
            
            if 'prompt' not in fieldnames:
                sys.exit("'prompt' column not found in CSV file")
            
            rows = list(reader)
            total_rows = len(rows)
            
        print(f"Found {total_rows} rows with columns: {fieldnames}")
        
        with open(dst, 'w', encoding='utf-8', newline='') as outfile:
            new_fieldnames = fieldnames + ['broken_type']
            writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
            writer.writeheader()
            
            prompt_types = ["word_order", "abbreviations", "keywords", "colloquial"]
            successful_count = 0
            
            for idx, row in enumerate(tqdm(rows, desc="Processing prompts")):
                prompt_text = row.get('prompt', '').strip()
                
                if not prompt_text:
                    print(f"Empty prompt at row {idx}, skipping", file=sys.stderr)
                    continue
                
                prompt_idx = idx % 4
                
                try:
                    broken_prompt = break_question(prompt_text, client, prompt_idx)
                    
                    new_row = row.copy()
                    new_row['prompt'] = broken_prompt
                    new_row['broken_type'] = prompt_types[prompt_idx]
                    
                    writer.writerow(new_row)
                    successful_count += 1
                    
                except Exception as e:
                    print(f"Skipping row {idx} due to error: {e}", file=sys.stderr)
                    continue
                
                time.sleep(0.1)
    
    except FileNotFoundError:
        sys.exit(f"Input file '{src}' not found")
    except Exception as e:
        sys.exit(f"Error processing file: {e}")
    
    print(f"Done! Processed {successful_count}/{total_rows} rows")
    print(f"Broken prompts saved to {dst}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Break CSV prompts into casual/broken versions")
    parser.add_argument("--input", "-i", default="oasst1_test.csv", help="Input CSV file")
    parser.add_argument("--output", "-o", default="oasst1_broken_test.csv", help="Output CSV file")
    
    args = parser.parse_args()
    main(args.input, args.output)