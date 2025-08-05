# Intent-Preserving Query Rewriting (IPQR)

<div align="center">
  <img src="https://github.com/dev-guardy/IPQR/blob/main/logo.png" alt="IPQR Logo" width="1000"/>
</div>

<div align="center">
  <h3>An Embedding-Based Framework for Large Language Models</h3>
</div>

This repository contains the implementation of **Intent-Preserving Query Rewriting (IPQR)**, a novel framework that transforms imperfect user queries into well-formed queries while maintaining the original user intent. Unlike existing methods that assume well-formed queries, IPQR is specifically designed for real-world user interactions with incomplete phrases, keyword-based searches, and grammatically imperfect statements.

## 📋 Table of Contents

- [Overview](#overview)  
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Pipeline Execution](#pipeline-execution)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

IPQR addresses the fundamental limitation that existing query rewriting methods assume well-formed, grammatically correct user queries. Our framework:

- **Handles Real-World Queries**: Processes imperfect queries with altered word orders, abbreviations, keyword fragments, and colloquial expressions
- **Preserves User Intent**: Uses triplet-trained embeddings to select optimal rewrites without over-modification
- **Achieves Superior Performance**: Outperforms existing single-round methods on realistic datasets while maintaining competitive performance on standard benchmarks
- **Reduces Training Cost**: Significantly lower computational requirements compared to reinforcement learning approaches

### Key Contributions

1. **Realistic Dataset Construction**: Transform existing benchmarks to incorporate real-world user query patterns
2. **Embedding-Driven Intent Preservation**: Novel single-round approach using triplet-trained embeddings
3. **Superior Practical Performance**: Better results on realistic datasets with reduced training time

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+
- NVIDIA GPUs with sufficient VRAM (recommended: A100 40GB+ for training)

### Setup

1. **Create virtual environment:**
   ```bash
   conda create --name ipqr python=3.8.0
   conda activate ipqr
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure access:**
   - Obtain HuggingFace access for Llama models
   - Set OpenAI API key for dataset generation
   - Configure WandB for training monitoring (optional)

## 📊 Dataset Preparation

### Standard Datasets

We evaluate on three established QA datasets:

- **K-QA Dataset**: Medical domain with FActScore methodology
- **TruthfulQA**: Multi-domain truthfulness evaluation (38 categories)  
- **OQA**: Conversational dialogue benchmark

### Realistic Dataset Construction

Based on analysis of real user query patterns, we transform standard datasets to include:

- **Altered Word Orders** (23%): Scrambled syntax, missing articles
- **Abbreviations & Acronyms** (27%): Texting shortcuts, shortened forms
- **Keyword-Based Queries** (26%): Fragmented search terms
- **Colloquial Expressions** (24%): Conversational, informal tone

## 🚀 Pipeline Execution

### Phase 1: Data Generation and Processing

#### 1. Generate Original Answers and Scores

```bash
# K-QA Dataset
python original_kqa.py --device_respond "cuda:0"

# TruthfulQA Dataset  
python original_tqa.py --device_respond "cuda:0" --device_judge_1 "cuda:1" --device_judge_2 "cuda:2"

# OQA Dataset
python original_oqa.py --device_respond "cuda:0" --device_judge "cuda:1"
```

#### 2. Generate Broken Questions (4 Types)

```bash
# K-QA Dataset
python datasets/K-QA/dataset/generate_broken_questions_kqa.py --openai_api_key "your_openai_api_key"

# TruthfulQA Dataset
python datasets/TruthfulQA/dataset/generate_broken_questions_tqa.py --openai_api_key "your_openai_api_key"

# OQA Dataset  
python datasets/OQA/dataset/generate_broken_questions_oqa.py --openai_api_key "your_openai_api_key"
```

#### 3. Generate IPQR Training Data

```bash
python generate_ipqr_data.py
```

This creates `data/pairs.jsonl` containing:
- Original and broken question pairs
- Ground truth answers and must-have information
- Train/validation splits

### Phase 2: Model Training

#### 4. Train Question Rewriter

```bash
cd ipqr
python generate_sft_data.py
```

Processes question pairs through:
- Llama-3.2-3B-Instruct rewriter model
- BGE-M3 embedding-based evaluation
- Generates `improved_pairs.jsonl`

#### 5. Create SFT Training Data

```bash
python create_training_data.py
```

Features:
- BGE-M3 similarity filtering (threshold: 0.8)
- Llama 3.2 format preparation
- Generates `training_data_llama_bge_filtered.jsonl`

#### 6. Fine-tune BGE-M3 Embedding Model

```bash
python bge_trainer.py \
    --cuda 0 \
    --epochs 6 \
    --batch_size 32 \
    --learning_rate 5e-6 \
    --wandb_project "bge-m3-finetuning" \
    --wandb_name "bge-m3-ipqr"
```

Training features:
- **Triplet Loss**: Ranks improved questions using composite scoring
- **Automatic Dataset Splitting**: 80% train, 10% validation, 10% test
- **Performance Monitoring**: Real-time accuracy and similarity metrics
- **Dual Format Output**: SentenceTransformer + HuggingFace compatible

## 🧪 Evaluation

### Test IPQR Performance

#### K-QA Evaluation

```bash
# Standard questions (clean, well-formed)
python test_kqa_ipqr.py \
    --device_judge "cuda:0" \
    --device_completer "cuda:1" \
    --device_respond "cuda:2" \
    --openai_api_key "your_openai_api_key"

# Broken questions (real-world scenario)  
python test_kqa_ipqr_r.py \
    --device_judge "cuda:0" \
    --device_completer "cuda:1" \
    --device_respond "cuda:2" \
    --openai_api_key "your_openai_api_key"
```

#### TruthfulQA Evaluation

```bash
# Standard questions
python test_tqa_ipqr.py \
    --device_judge_1 "cuda:0" \
    --device_judge_2 "cuda:1" \
    --device_completer "cuda:2" \
    --device_respond "cuda:3" \
    --openai_api_key "your_openai_api_key"

# Broken questions (real-world scenario)
python test_tqa_ipqr_r.py \
    --device_judge_1 "cuda:0" \
    --device_judge_2 "cuda:1" \
    --device_completer "cuda:2" \
    --device_respond "cuda:3" \
    --openai_api_key "your_openai_api_key"
```

#### OASST1QA Evaluation

```bash
# Standard questions
python test_oqa_ipqr.py \
    --device_judge "cuda:0" \
    --device_completer "cuda:1" \
    --device_respond "cuda:2" \
    --openai_api_key "your_openai_api_key"

# Broken questions (real-world scenario)
python test_oqa_ipqr_r.py \
    --device_judge "cuda:0" \
    --device_completer "cuda:1" \
    --device_respond "cuda:2" \
    --openai_api_key "your_openai_api_key"
```

### Evaluation Metrics

- **K-QA**: Comprehensiveness ($S_{comp}$ ↑) and Contradiction ($S_{cont}$ ↓)
- **TruthfulQA**: Truthfulness ($S_{truth}$ ↑), Informativeness ($S_{info}$ ↑)  
- **OQA**: Human Preference Score ($S_{pref}$ ↑)

### Real-World Testing Significance

**Broken Questions Testing** (`*_r.py` scripts) represents authentic user scenarios:
- Grammatical errors and casual language
- Search query-style fragmented input
- Abbreviations and colloquial expressions
- This evaluation demonstrates IPQR's effectiveness for real-world deployment

## 📈 Results

### Performance Highlights

- **Competitive on Standard Datasets**: Matches state-of-the-art DPO-based methods
- **Superior on Realistic Datasets**: Significantly outperforms existing approaches on broken queries
- **Cross-Model Generalization**: Effective across Llama, Mistral, Zephyr, Gemma, and GPT models
- **Training Efficiency**: 3x faster training with 4x lower GPU memory requirements

### Key Findings

| Method | Training Time | GPU Memory | Standard Dataset | Realistic Dataset |
|--------|---------------|------------|------------------|-------------------|
| DPO (Baseline) | 16 hours | 80GB | Strong | Weak |
| **IPQR (Ours)** | **3 hours** | **20GB** | **Strong** | **Strong** |

## 🏗️ Framework Architecture

```
Input Query (Original/Broken) 
    ↓
Query Rewriter (Llama-3.2-3B) → Generate n=3 candidates
    ↓
BGE-M3 Embedding Selection → Select best intent-preserving rewrite
    ↓  
Target LLM → Generate final answer
```

### Training Components

1. **Better Query Generation**: Creates improved questions with higher answer quality scores
2. **Rewriter Training**: LoRA fine-tuning of Llama-3.2-3B to generate candidate rewrites
3. **Embedding Training**: Triplet loss training of BGE-M3 for intent preservation
4. **Selection Mechanism**: Cosine similarity-based selection preserving user intent

## 📁 Repository Structure

```
ipqr/
├── datasets/                  # Standard QA datasets
│   ├── K-QA/
│   │   └── dataset/
│   │       ├── questions_w_answers.jsonl
│   │       └── generate_broken_questions_kqa.py
│   ├── TruthfulQA/
│   │   └── dataset/
│   │       └── generate_broken_questions_tqa.py
│   └── OQA/
│       └── dataset/
│           └── generate_broken_questions_oqa.py
├── data/                      # Generated data files
│   ├── original/              # Original answers and scores
│   ├── rewrite/              # Rewritten questions
│   └── dpo/                  # Training data
├── ipqr/                      # Core IPQR implementation
│   ├── generate_sft_data.py
│   ├── create_training_data.py
│   └── bge_trainer.py
├── test_results/             # Evaluation outputs
├── intermediate_results/     # Processing intermediates
├── llms/                    # LLM interface modules
├── generate_ipqr_data.py    # IPQR data generation
├── test_kqa_ipqr.py         # K-QA standard evaluation
├── test_kqa_ipqr_r.py       # K-QA broken evaluation
├── test_tqa_ipqr.py         # TruthfulQA standard evaluation  
├── test_tqa_ipqr_r.py       # TruthfulQA broken evaluation
├── test_oqa_ipqr.py         # OQA standard evaluation
├── test_oqa_ipqr_r.py       # OQA broken evaluation
├── original_kqa.py          # K-QA original processing
├── original_tqa.py          # TruthfulQA original processing
├── original_oqa.py          # OQA original processing
└── requirements.txt
```
### Referenced Work

Our work builds upon and compares with:

```
@article{chen2025putting,
  title={Putting People in LLMs' Shoes: Generating Better Answers via Question Rewriter},
  author={Chen, Zheng and others},
  journal={arXiv preprint arXiv:2408.10573},
  year={2025}
}

@article{kong2024prewrite,
  title={PRewrite: Prompt Rewriting with Reinforcement Learning},
  author={Kong, Weize and others},
  journal={ICML 2024},
  year={2024}
}
```

---

**Note**: This implementation requires significant computational resources for training. For reproduction, we recommend using high-end GPUs (A100 40GB+) and ensuring sufficient disk space for intermediate data storage.
