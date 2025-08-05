import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import random
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import argparse
import wandb
from datetime import datetime
import time
import shutil

def setup_device(cuda_device: int = None):
    if cuda_device is not None:
        if torch.cuda.is_available():
            device = f"cuda:{cuda_device}"
            torch.cuda.set_device(cuda_device)
            print(f"Using CUDA device: {cuda_device}")
            print(f"GPU Name: {torch.cuda.get_device_name(cuda_device)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(cuda_device).total_memory / 1024**3:.1f} GB")
        else:
            print("CUDA not available, using CPU")
            device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"Using default CUDA device: {torch.cuda.current_device()}")
        else:
            print("Using CPU")
    
    return device

class SimpleTripletEvaluator(TripletEvaluator):
    def __init__(self, anchors, positives, negatives, name='', use_wandb=False):
        super().__init__(anchors, positives, negatives, name=name)
        self.use_wandb = use_wandb
        
    def __call__(self, model, output_path, epoch, steps):
        score = super().__call__(model, output_path, epoch, steps)
        
        if isinstance(score, dict):
            accuracy = score.get('accuracy', 0.0)
            print(f"[Validation] Epoch {epoch}, Steps {steps}")
            print(f"[Validation] Triplet Accuracy: {accuracy:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "val/accuracy": accuracy,
                    "val/epoch": epoch,
                    "epoch": epoch,
                    "step": steps
                }, step=steps)
            
            return accuracy
        else:
            print(f"[Validation] Epoch {epoch}, Steps {steps}")
            print(f"[Validation] Triplet Accuracy: {score:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "val/accuracy": score,
                    "val/epoch": epoch,
                    "epoch": epoch,
                    "step": steps
                }, step=steps)
            
            return score

class BGEDatasetCreator:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.triplets = []
        
    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def calculate_similarity(self, query: str, answer: str) -> float:
        query_embedding = self.model.encode([query])
        answer_embedding = self.model.encode([answer])
        similarity = cosine_similarity(query_embedding, answer_embedding)[0][0]
        return float(similarity)
    
    def create_triplet_data(self, data: List[Dict]) -> List[Tuple[str, str, str]]:
        triplets = []
        print("Creating triplet data...")

        for item in tqdm(data):
            if item.get("successful_attempts", 0) < 3:
                continue

            original_question = item.get("question")
            broken_question = item.get("broken")
            improvements = item.get("all_improvements", [])
            if len(improvements) < 2:
                continue

            improved_list = [
                {"question": imp["question"], "score": imp["score"]}
                for imp in improvements
                if "question" in imp and "score" in imp
            ]
            if len(improved_list) < 2:
                continue

            for anchor in [original_question, broken_question]:
                if not anchor:
                    continue

                scored = []
                for imp in improved_list:
                    sim = self.calculate_similarity(anchor, imp["question"])
                    combined = 0.5 * imp["score"] + 0.5 * sim
                    scored.append({"question": imp["question"], "combined": combined})

                scored.sort(key=lambda x: x["combined"], reverse=True)

                for i in range(len(scored)):
                    for j in range(len(scored)):
                        if i == j:
                            continue
                        if scored[i]["combined"] > scored[j]["combined"]:
                            positive = scored[i]["question"]
                            negative = scored[j]["question"]
                            triplets.append((anchor, positive, negative))

        print(f"Generated {len(triplets)} triplets")
        return triplets
    
    def split_dataset(self, triplets: List[Tuple[str, str, str]], 
                     train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, 
                     test_ratio: float = 0.1):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        random.shuffle(triplets)
        total_size = len(triplets)
        
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_data = triplets[:train_size]
        val_data = triplets[train_size:train_size + val_size]
        test_data = triplets[train_size + val_size:]
        
        print(f"Dataset split:")
        print(f"  Train: {len(train_data)} ({len(train_data)/total_size*100:.1f}%)")
        print(f"  Val:   {len(val_data)} ({len(val_data)/total_size*100:.1f}%)")
        print(f"  Test:  {len(test_data)} ({len(test_data)/total_size*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def save_split_data(self, train_data, val_data, test_data, base_path: str = "dataset"):
        os.makedirs(base_path, exist_ok=True)
        
        datasets = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
        
        for split_name, data in datasets.items():
            file_path = os.path.join(base_path, f"{split_name}_triplets.jsonl")
            with open(file_path, 'w', encoding='utf-8') as f:
                for anchor, positive, negative in data:
                    json.dump({
                        "anchor": anchor,
                        "positive": positive,
                        "negative": negative
                    }, f, ensure_ascii=False)
                    f.write('\n')
            print(f"Saved {len(data)} {split_name} triplets to {file_path}")
    
    def load_split_data(self, base_path: str = "dataset"):
        datasets = {}
        for split_name in ["train", "val", "test"]:
            file_path = os.path.join(base_path, f"{split_name}_triplets.jsonl")
            triplets = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    triplets.append((data["anchor"], data["positive"], data["negative"]))
            datasets[split_name] = triplets
            print(f"Loaded {len(triplets)} {split_name} triplets")
        return datasets["train"], datasets["val"], datasets["test"]
    
    def create_input_examples(self, triplets: List[Tuple[str, str, str]]) -> List[InputExample]:
        examples = []
        for anchor, positive, negative in triplets:
            examples.append(InputExample(texts=[anchor, positive, negative]))
        return examples

class BGEEvaluator:
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.model_name = model_name_or_path
        self.device = device
        
    def evaluate_triplets(self, test_triplets: List[Tuple[str, str, str]], 
                         sample_size: int = None) -> Dict[str, float]:
        if sample_size:
            test_triplets = random.sample(test_triplets, min(sample_size, len(test_triplets)))
        
        correct = 0
        total = len(test_triplets)
        positive_similarities = []
        negative_similarities = []
        margins = []
        
        print(f"Evaluating model on {total} triplets...")
        for anchor, positive, negative in tqdm(test_triplets):
            anchor_emb = self.model.encode([anchor])
            positive_emb = self.model.encode([positive])
            negative_emb = self.model.encode([negative])
            
            pos_sim = cosine_similarity(anchor_emb, positive_emb)[0][0]
            neg_sim = cosine_similarity(anchor_emb, negative_emb)[0][0]
            
            positive_similarities.append(pos_sim)
            negative_similarities.append(neg_sim)
            margins.append(pos_sim - neg_sim)
            
            if pos_sim > neg_sim:
                correct += 1
        
        accuracy = correct / total
        avg_pos_sim = np.mean(positive_similarities)
        avg_neg_sim = np.mean(negative_similarities)
        avg_margin = np.mean(margins)
        
        results = {
            "accuracy": accuracy,
            "avg_positive_similarity": avg_pos_sim,
            "avg_negative_similarity": avg_neg_sim,
            "avg_margin": avg_margin,
            "total_samples": total,
            "correct_predictions": correct
        }
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, float], prefix: str = ""):
        print(f"\n=== {prefix} Evaluation Results ===")
        print(f"Model: {self.model_name}")
        print(f"Accuracy: {results['accuracy']:.4f} ({results['correct_predictions']}/{results['total_samples']})")
        print(f"Average Positive Similarity: {results['avg_positive_similarity']:.4f}")
        print(f"Average Negative Similarity: {results['avg_negative_similarity']:.4f}")
        print(f"Average Margin (Pos - Neg): {results['avg_margin']:.4f}")

class SimpleTripletLoss(losses.TripletLoss):
    def __init__(self, model, distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE, 
                 triplet_margin=0.5, use_wandb=False):
        super().__init__(model, distance_metric, triplet_margin)
        self.use_wandb = use_wandb
        self.step_count = 0
        self.recent_losses = []
        
    def forward(self, sentence_features, labels):
        loss = super().forward(sentence_features, labels)
        
        self.step_count += 1
        loss_value = loss.item()
        self.recent_losses.append(loss_value)
        
        if self.step_count % 5 == 0:
            avg_loss = np.mean(self.recent_losses[-10:])
            print(f"[Training] Step {self.step_count}, Loss: {loss_value:.4f}, Avg Loss (last 10): {avg_loss:.4f}")
        
            if self.use_wandb:
                wandb.log({
                    "train/loss": loss_value,
                    "train/avg_loss": avg_loss,
                    "step": self.step_count
                }, step=self.step_count)
        
        return loss

class BGETrainer:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda", use_wandb: bool = False):
        self.model_name = model_name
        self.model = None
        self.device = device
        self.use_wandb = use_wandb
        
    def load_model(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"Loaded model: {self.model_name} on device: {self.device}")
        
    def prepare_dataloader(self, examples: List[InputExample], batch_size: int = 16) -> DataLoader:
        return DataLoader(examples, shuffle=True, batch_size=batch_size)
    
    def save_as_huggingface_model(self, sentence_transformer_path: str, output_path: str):
        print(f"Converting SentenceTransformer model to Hugging Face format...")
        
        try:
            if self.model is not None:
                transformer_model = self.model._modules['0']
            else:
                config_path = os.path.join(sentence_transformer_path, "config.json")
                modules_config_path = os.path.join(sentence_transformer_path, "modules.json")
                
                if not os.path.exists(config_path) or not os.path.exists(modules_config_path):
                    print(f"Warning: Required config files not found in {sentence_transformer_path}")
                    print("Skipping Hugging Face format conversion...")
                    return sentence_transformer_path
                
                st_model = SentenceTransformer(sentence_transformer_path, device='cpu')
                transformer_model = st_model._modules['0']
            
            os.makedirs(output_path, exist_ok=True)
            
            transformer_model.auto_model.save_pretrained(output_path)
            transformer_model.tokenizer.save_pretrained(output_path)
            
            config_path = os.path.join(output_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if 'model_type' not in config:
                    config['model_type'] = 'bert'
                    
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            
            model_info = {
                "model_type": "bge-m3-finetuned",
                "sentence_transformer_path": sentence_transformer_path,
                "base_model": self.model_name,
                "conversion_successful": True
            }
            
            with open(os.path.join(output_path, "finetuning_info.json"), 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            print(f"Model successfully converted and saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error during Hugging Face conversion: {str(e)}")
            print(f"Returning original SentenceTransformer path: {sentence_transformer_path}")
            return sentence_transformer_path

    def train(self, 
            train_examples: List[InputExample],
            val_examples: List[InputExample] = None,
            output_path: str = "./bge-m3-finetuned",
            hf_output_path: str = "./bge-m3-finetuned-hf",
            epochs: int = 5,
            batch_size: int = 16,
            learning_rate: float = 1e-5,
            warmup_steps: int = 100):
        if self.model is None:
            self.load_model()
        
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(hf_output_path, exist_ok=True)
            
        train_dataloader = self.prepare_dataloader(train_examples, batch_size)
        
        train_loss = SimpleTripletLoss(model=self.model, use_wandb=self.use_wandb, triplet_margin=0.5)
        
        evaluator = None
        if val_examples:
            eval_size = min(300, len(val_examples))
            anchors = [example.texts[0] for example in val_examples[:eval_size]]
            positives = [example.texts[1] for example in val_examples[:eval_size]]
            negatives = [example.texts[2] for example in val_examples[:eval_size]]
            evaluator = SimpleTripletEvaluator(
                anchors, positives, negatives, 
                name='validation',
                use_wandb=self.use_wandb
            )
        
        total_steps = len(train_dataloader) * epochs
        evaluation_steps = 100
        
        print(f"\n=== Training Started ===")
        print(f"Train examples: {len(train_examples)}")
        print(f"Batch size: {batch_size} -> Steps per epoch: {len(train_dataloader)}")
        print(f"Total steps: {total_steps}")
        print(f"Evaluation every: {evaluation_steps} steps")
        print(f"Epochs: {epochs}, Learning rate: {learning_rate}")
        
        if self.use_wandb:
            wandb.log({
                "train/total_steps": total_steps,
                "train/steps_per_epoch": len(train_dataloader),
                "train/total_examples": len(train_examples),
                "config/batch_size": batch_size,
                "config/learning_rate": learning_rate,
                "config/epochs": epochs,
                "config/evaluation_steps": evaluation_steps
            })
        
        start_time = time.time()
        
        try:
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                evaluator=evaluator,
                evaluation_steps=evaluation_steps,
                warmup_steps=warmup_steps,
                output_path=output_path,
                optimizer_params={
                    'lr': learning_rate,
                    'weight_decay': 0.01,
                    'eps': 1e-6
                },
                save_best_model=True,
                show_progress_bar=True
            )
            
            end_time = time.time()
            training_duration = end_time - start_time
            
            print(f"\n=== Training Completed ===")
            print(f"Training time: {training_duration/60:.2f} minutes")
            print(f"SentenceTransformer model saved to: {output_path}")
            
            if self.use_wandb:
                wandb.log({
                    "train/training_duration_minutes": training_duration/60,
                    "train/training_completed": True
                })
            
            final_model_path = self.save_as_huggingface_model(output_path, hf_output_path)
            
            return final_model_path
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            if self.use_wandb:
                wandb.log({"train/error": str(e)})
            return output_path

def main():
    parser = argparse.ArgumentParser(description='Fine-tune BGE-M3 model')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number')
    parser.add_argument('--data_file', type=str, default='improved_pairs.jsonl', help='Path to the input JSONL data file')
    parser.add_argument('--epochs', type=int, default=6, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--wandb_project', type=str, default='bge-m3-finetuning', help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default='bge-m3-finetuning', help='WandB experiment name')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    device = setup_device(args.cuda)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    use_wandb = not args.disable_wandb
    if use_wandb:
        experiment_name = args.wandb_name or f"bge-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=experiment_name,
            config={
                "model_name": "BAAI/bge-m3",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "device": device,
                "data_file": args.data_file,
                "warmup_steps": 100,
                "triplet_margin": 0.5,
                "evaluation_steps": 100
            }
        )
        print(f"WandB initialized: {args.wandb_project}/{experiment_name}")

    print(f"\n=== Configuration ===")
    print(f"Device: {device}")
    print(f"Data file: {args.data_file}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"WandB logging: {'Enabled' if use_wandb else 'Disabled'}")

    try:
        if not os.path.exists("dataset/train_triplets.jsonl"):
            print("\n=== Creating Dataset ===")
            dataset_creator = BGEDatasetCreator(device=device)
            data = dataset_creator.load_jsonl_data(args.data_file)
            print(f"Loaded {len(data)} items from dataset")
            
            if use_wandb:
                wandb.log({"dataset/total_items": len(data)})
            
            triplets = dataset_creator.create_triplet_data(data)
            train_data, val_data, test_data = dataset_creator.split_dataset(
                triplets, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
            )
            dataset_creator.save_split_data(train_data, val_data, test_data)
            
            if use_wandb:
                wandb.log({
                    "dataset/total_triplets": len(triplets),
                    "dataset/train_size": len(train_data),
                    "dataset/val_size": len(val_data),
                    "dataset/test_size": len(test_data)
                })
        else:
            print("\n=== Loading Existing Dataset ===")
            dataset_creator = BGEDatasetCreator(device=device)
            train_data, val_data, test_data = dataset_creator.load_split_data()
            
            if use_wandb:
                wandb.log({
                    "dataset/train_size": len(train_data),
                    "dataset/val_size": len(val_data),
                    "dataset/test_size": len(test_data),
                    "dataset/total_triplets": len(train_data) + len(val_data) + len(test_data)
                })

        print("\n=== Evaluating Pre-trained Model ===")
        evaluator = BGEEvaluator("BAAI/bge-m3", device=device)
        pre_train_results = evaluator.evaluate_triplets(test_data, sample_size=200)
        evaluator.print_evaluation_results(pre_train_results, "Pre-training")
        
        if use_wandb:
            wandb.log({
                "baseline/accuracy": pre_train_results["accuracy"],
                "baseline/avg_positive_similarity": pre_train_results["avg_positive_similarity"],
                "baseline/avg_negative_similarity": pre_train_results["avg_negative_similarity"],
                "baseline/avg_margin": pre_train_results["avg_margin"],
                "baseline/total_samples": pre_train_results["total_samples"]
            })

        print(f"\n=== Starting Fine-tuning ===")
        train_examples = dataset_creator.create_input_examples(train_data)
        val_examples = dataset_creator.create_input_examples(val_data)

        trainer = BGETrainer(device=device, use_wandb=use_wandb)
        final_model_path = trainer.train(
            train_examples=train_examples,
            val_examples=val_examples,
            output_path="./bge-m3-medical-qa-finetuned",
            hf_output_path="./bge-m3-medical-qa-finetuned-hf",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=100
        )

        print("\n=== Evaluating Fine-tuned Model ===")
        try:
            evaluator = BGEEvaluator("./bge-m3-medical-qa-finetuned", device=device)
            post_train_results = evaluator.evaluate_triplets(test_data, sample_size=200)
            evaluator.print_evaluation_results(post_train_results, "Post-training")

            if use_wandb:
                wandb.log({
                    "final/accuracy": post_train_results["accuracy"],
                    "final/avg_positive_similarity": post_train_results["avg_positive_similarity"],
                    "final/avg_negative_similarity": post_train_results["avg_negative_similarity"],
                    "final/avg_margin": post_train_results["avg_margin"],
                    "final/total_samples": post_train_results["total_samples"],
                    "improvement/accuracy": post_train_results["accuracy"] - pre_train_results["accuracy"],
                    "improvement/margin": post_train_results["avg_margin"] - pre_train_results["avg_margin"],
                    "improvement/accuracy_pct": ((post_train_results["accuracy"] - pre_train_results["accuracy"]) / pre_train_results["accuracy"]) * 100,
                    "improvement/margin_pct": ((post_train_results["avg_margin"] - pre_train_results["avg_margin"]) / pre_train_results["avg_margin"]) * 100 if pre_train_results["avg_margin"] != 0 else 0
                })

            print("\n=== Performance Comparison ===")
            metrics = ["accuracy", "avg_positive_similarity", "avg_negative_similarity", "avg_margin"]
            
            print(f"{'Metric':<30} {'Pre-training':<15} {'Post-training':<15} {'Improvement':<15}")
            print("-" * 75)
            
            comparison_results = {}
            for metric in metrics:
                pre_val = pre_train_results[metric]
                post_val = post_train_results[metric]
                improvement = post_val - pre_val
                improvement_pct = (improvement / pre_val) * 100 if pre_val != 0 else 0
                
                print(f"{metric:<30} {pre_val:<15.4f} {post_val:<15.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
                
                comparison_results[f"comparison/{metric}_pre"] = pre_val
                comparison_results[f"comparison/{metric}_post"] = post_val
                comparison_results[f"comparison/{metric}_improvement"] = improvement
                comparison_results[f"comparison/{metric}_improvement_pct"] = improvement_pct
            
            if use_wandb:
                wandb.log(comparison_results)
                
        except Exception as eval_error:
            print(f"Error during final evaluation: {str(eval_error)}")
            print("Training completed but final evaluation failed.")
            if use_wandb:
                wandb.log({"final/evaluation_error": str(eval_error)})

        print(f"\n=== Training Complete ===")
        print(f"Fine-tuned model saved to: {final_model_path}")
        if use_wandb:
            print(f"View training results at: {wandb.run.url}")
            wandb.log({
                "status/training_completed": True,
                "status/final_model_path": final_model_path
            })

    except Exception as e:
        print(f"Error during training: {str(e)}")
        if use_wandb:
            wandb.log({
                "status/error": str(e),
                "status/training_completed": False
            })
        raise
    
    finally:
        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()