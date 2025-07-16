#!/usr/bin/env python3

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer
import numpy as np
from datetime import datetime

# 设置使用hf-mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class MinimalDataSetup:
    def __init__(self, tokenizer_name: str = "01-ai/Yi-6B"):

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.cache_dir = Path("./dataset_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def check_local_cache(self, dataset_name: str, config: Optional[str] = None) -> bool:

        cache_path = self.cache_dir / dataset_name.replace("/", "___")
        if config:
            cache_path = cache_path / config
        

        if cache_path.exists():
            data_files = list(cache_path.rglob("*.arrow")) + list(cache_path.rglob("*.parquet"))
            if data_files:
                return True
        return False
    
    def load_from_local_cache(self, dataset_name: str, config: Optional[str] = None, 
                             split: str = "train", max_samples: Optional[int] = None):
        try:

            load_kwargs = {
                'cache_dir': str(self.cache_dir),
                'download_mode': 'reuse_cache_if_exists', 
                'trust_remote_code': True
            }
            
            if config:
                load_kwargs['name'] = config
            
            if max_samples:
                dataset = load_dataset(
                    dataset_name,
                    split=f"{split}[:{max_samples}]",
                    **load_kwargs
                )
            else:
                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    **load_kwargs
                )
            return dataset
            
        except Exception as e:
            return None
    
    def download_arxiv_minimal(self, forget_size=400, validate_size=200):
        
        if self.check_local_cache("armanc/scientific_papers", "arxiv"):
            try:
                total_needed = forget_size + validate_size + 100
                dataset = self.load_from_local_cache(
                    "armanc/scientific_papers", 
                    "arxiv", 
                    "train", 
                    total_needed
                )
                
                if dataset is None:
                    return self.create_backup_arxiv_data(forget_size, validate_size)
                
            except Exception as e:
                return self.create_backup_arxiv_data(forget_size, validate_size)
        else:
            try:
                total_needed = forget_size + validate_size + 100
                dataset = load_dataset(
                    "armanc/scientific_papers", 
                    "arxiv",
                    split=f"train[:{total_needed}]",
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True
                )
                
            except Exception as e:
                return self.create_backup_arxiv_data(forget_size, validate_size)
        
        forget_texts = []
        validate_texts = []
        
        for i, paper in enumerate(dataset):
            if len(forget_texts) >= forget_size and len(validate_texts) >= validate_size:
                break
            
            text_parts = []
            if paper.get('title'):
                text_parts.append(f"Title: {paper['title']}")
            if paper.get('abstract'):
                text_parts.append(f"Abstract: {paper['abstract']}")
            if paper.get('article'):
                article = paper['article'][:1500] if len(paper['article']) > 1500 else paper['article']
                text_parts.append(f"Article: {article}")
            
            if text_parts:
                full_text = "\n\n".join(text_parts)
                

                if len(forget_texts) < forget_size:
                    forget_texts.append(full_text)

                elif len(validate_texts) < validate_size:
                    validate_texts.append(full_text)
        
        return forget_texts, validate_texts
    
    def create_backup_arxiv_data(self, forget_size=400, validate_size=200):

        academic_templates = [
            "Deep learning has revolutionized artificial intelligence by enabling machines to learn complex patterns from large datasets. Neural networks with multiple layers can automatically extract hierarchical features, leading to breakthroughs in computer vision, natural language processing, and speech recognition.",
            
            "Transformer architectures have transformed natural language processing through the introduction of self-attention mechanisms. These models can process sequences in parallel, capturing long-range dependencies more effectively than recurrent neural networks.",
            
            "Convolutional neural networks excel at image recognition tasks by applying learnable filters that detect local features such as edges, textures, and shapes. The hierarchical structure enables the network to build complex representations from simple patterns.",
            
            "Reinforcement learning trains agents to make optimal decisions through trial-and-error interactions with an environment. The agent receives rewards or penalties based on its actions, learning to maximize cumulative rewards over time.",
            
            "Generative adversarial networks consist of two neural networks competing against each other: a generator that creates fake data and a discriminator that tries to distinguish real from fake data.",
            
            "Transfer learning leverages pre-trained models to solve new tasks with limited data. By fine-tuning models that have already learned general features from large datasets, we can achieve better performance on specific tasks.",
            
            "Attention mechanisms in neural networks allow models to selectively focus on relevant parts of the input when making predictions. This capability has proven essential for tasks involving long sequences.",
            
            "Regularization techniques prevent overfitting in deep neural networks by introducing constraints or noise during training. Methods such as dropout, batch normalization, and weight decay help models generalize better.",
            
            "Unsupervised learning discovers hidden patterns in data without labeled examples. Techniques such as clustering, dimensionality reduction, and autoencoder networks can extract meaningful representations from raw data.",
            
            "Optimization algorithms for neural networks have evolved from simple gradient descent to sophisticated methods like Adam, RMSprop, and AdaGrad. These adaptive learning rate methods adjust the step size for each parameter individually."
        ]
        

        forget_texts = []
        for i in range(forget_size):
            template = academic_templates[i % len(academic_templates)]
            paper_text = f"Title: Advanced Research in Machine Learning - Paper {i+1}\n\nAbstract: {template}\n\nKeywords: machine learning, artificial intelligence, deep learning, neural networks\n\nIntroduction: This paper presents novel contributions to the field of artificial intelligence, specifically focusing on advanced machine learning techniques and their applications."
            forget_texts.append(paper_text)
        

        validate_texts = []
        for i in range(validate_size):
            template = academic_templates[i % len(academic_templates)]
            paper_text = f"Research Article {i+1}: {template}\n\nThis study demonstrates significant improvements in model performance and provides theoretical insights into the underlying mechanisms."
            validate_texts.append(paper_text)
        
        return forget_texts, validate_texts
    
    def download_github_minimal(self, forget_size=400, validate_size=200):
        if self.check_local_cache("bigcode/the-stack"):
            try:

                dataset = load_dataset(
                    "bigcode/the-stack", 
                    data_dir="data/python",
                    split="train",
                    cache_dir=str(self.cache_dir),
                    streaming=True,
                    download_mode='reuse_cache_if_exists'
                )

                
            except Exception as e:

                return self.create_backup_github_data(forget_size, validate_size)
        else:

            try:
                dataset = load_dataset(
                    "bigcode/the-stack", 
                    data_dir="data/python",
                    split="train",
                    cache_dir=str(self.cache_dir),
                    streaming=True
                )
                
            except Exception as e:
                return self.create_backup_github_data(forget_size, validate_size)
        

        forget_texts = []
        validate_texts = []
        count = 0
        max_attempts = (forget_size + validate_size) * 3  
        
        for item in dataset:
            count += 1
            if count > max_attempts or (len(forget_texts) >= forget_size and len(validate_texts) >= validate_size):
                break
            
            if count % 500 == 0:
                print(f"forget={len(forget_texts)}, validate={len(validate_texts)}")
            
            if 'content' in item and item['content']:
                code_content = item['content'].strip()
                

                if (len(code_content) > 200 and 
                    len(code_content) < 3000 and 
                    code_content.count('\n') > 10 and
                    ('def ' in code_content or 'class ' in code_content) and
                    'import ' in code_content):
                    

                    file_path = item.get('max_stars_repo_path', 'unknown_file.py')
                    repo_name = item.get('max_stars_repo_name', 'unknown_repo')
                    
                    formatted_code = f"# File: {file_path}\n# Repository: {repo_name}\n\n{code_content}"
                    

                    if len(forget_texts) < forget_size:
                        forget_texts.append(formatted_code)
                    elif len(validate_texts) < validate_size:
                        validate_texts.append(formatted_code)
        

        if len(forget_texts) < forget_size or len(validate_texts) < validate_size:
            backup_forget, backup_validate = self.create_backup_github_data(
                forget_size - len(forget_texts), 
                validate_size - len(validate_texts)
            )
            forget_texts.extend(backup_forget)
            validate_texts.extend(backup_validate)
        
        return forget_texts, validate_texts

    
def load_and_preprocess_data(file_path):
    """Load and preprocess dataset for machine learning."""
    data = pd.read_csv(file_path)
    data = data.dropna()
    scaler = StandardScaler()
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        return x''',
            
            '''# File: algorithms/sorting.py
# Repository: algorithms-collection

def quicksort(arr):
    """Implement quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)''',
            
            '''# File: web/flask_app.py
# Repository: web-application

from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users."""
    conn = sqlite3.connect('database.db')
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    return jsonify([dict(user) for user in users])''',
            
            '''# File: analysis/visualization.py
# Repository: data-science-tools

import matplotlib.pyplot as plt
import seaborn as sns

def create_heatmap(data, title="Correlation Matrix"):
    """Create correlation heatmap."""
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()'''
        ]
        

        forget_texts = []
        for i in range(forget_size):
            template = code_templates[i % len(code_templates)]
            code_text = template.replace("# Repository: ", f"# Repository: project-{i+1}-")
            forget_texts.append(code_text)
        

        validate_texts = []
        for i in range(validate_size):
            template = code_templates[i % len(code_templates)]
            code_text = template.replace("# Repository: ", f"# Repository: validation-{i+1}-")
            validate_texts.append(code_text)
        
        return forget_texts, validate_texts

    def create_retain_data(self, retain_size=300):

        general_knowledge = [
            "The Amazon rainforest is the world's largest tropical rainforest, covering approximately 5.5 million square kilometers across nine countries in South America. It plays a crucial role in global climate regulation and is home to countless species of plants and animals.",
            
            "Paris, the capital of France, is one of the world's most visited cities, known for its art, fashion, gastronomy, and culture. The city features iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
            
            "Mount Everest, located in the Himalayas on the border between Nepal and Tibet, stands at 8,848 meters above sea level, making it the world's highest mountain peak. It has been a challenging destination for mountaineers since the first successful ascent in 1953.",
            
            "The Renaissance was a period of European cultural, artistic, political, and economic rebirth following the Middle Ages. It is generally described as taking place from the 14th century to the 17th century, promoting the rediscovery of classical philosophy, literature, and art.",
            
            "Classical music encompasses a broad span of time from roughly the 11th century to the present day. The term 'classical music' is often used to refer to the formal musical tradition of the Western world, considered to be distinct from Western folk music or popular music traditions.",
            
            "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south and is bounded by the continents of Asia and Australia in the west and the Americas in the east.",
            
            "Ancient Egypt was a civilization of ancient Africa, concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC with the political unification of Upper and Lower Egypt.",
            
            "Photography is the art, application, and practice of creating durable images by recording light or other electromagnetic radiation. The word photography comes from the Greek words 'photos' meaning light and 'graphos' meaning drawing, literally meaning drawing with light.",
            
            "Cuisine is a characteristic style of cooking practices and traditions, often associated with a specific region, country, or culture. Different cuisines involve certain cooking methods, local ingredients, and cultural preferences that have been passed down through generations.",
            
            "Space exploration is the use of astronomy and space technology to explore outer space. While the exploration of space is carried out mainly by astronomers with telescopes, its physical exploration though is conducted both by unmanned robotic space probes and human spaceflight.",
            
            "Marine biology is the scientific study of the behavior, physiology, classification, and other aspects of marine organisms, as well as how organisms interact with each other and the environment. It is a vast field that encompasses many subdisciplines.",
            
            "Architecture is both the process and the product of planning, designing, and constructing buildings or other structures. Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art.",
            
            "Archaeology is the study of human activity through the recovery and analysis of material culture. The archaeological record consists of artifacts, architecture, biofacts or ecofacts, sites, and cultural landscapes.",
            
            "Meteorology is a branch of the atmospheric sciences which includes atmospheric chemistry and atmospheric physics, with a major focus on weather forecasting. The study of meteorology dates back millennia, though significant progress in meteorology did not occur until the 18th century.",
            
            "Geology is an earth science concerned with the solid Earth, the rocks of which it is composed, and the processes by which they change over time. Geology can also include the study of the solid features of any terrestrial planet or natural satellite such as Mars or the Moon."
        ]
        

        retain_texts = []
        for i in range(retain_size):
            base_text = general_knowledge[i % len(general_knowledge)]
            retain_text = f"General Knowledge Article {i+1}: {base_text}"
            retain_texts.append(retain_text)
        return retain_texts
    
    def tokenize_and_create_dataset(self, texts, max_length=512):
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        dataset_data = []
        for i in range(len(texts)):
            dataset_data.append({
                'text': texts[i],
                'input_ids': tokenized['input_ids'][i].tolist(),
                'attention_mask': tokenized['attention_mask'][i].tolist(),
                'labels': tokenized['input_ids'][i].tolist()
            })
        
        return dataset_data
    
    def save_scenario_data(self, scenario_name, forget_texts, retain_texts, validate_texts, output_dir="./preprocessed_data"):
        
        scenario_dir = Path(output_dir) / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        

        splits = {
            'forget': forget_texts,
            'retain': retain_texts,
            'validate': validate_texts
        }
        
        for split_name, texts in splits.items():
            if texts:
                dataset_data = self.tokenize_and_create_dataset(texts)
                
                file_path = scenario_dir / f"{split_name}_data.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset_data, f, indent=2, ensure_ascii=False)
                

        metadata = {
            'scenario': scenario_name,
            'created_at': datetime.now().isoformat(),
            'splits': {name: len(texts) for name, texts in splits.items() if texts},
            'tokenizer': self.tokenizer.name_or_path,
            'description': f"{scenario_name} scenario with forget/retain/validate splits"
        }
        
        metadata_path = scenario_dir / "dataset_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    
    def setup_minimal_data(self, 
                          forget_size=400, 
                          retain_size=300, 
                          validate_size=200,
                          output_dir="./preprocessed_data"):


        retain_texts = self.create_retain_data(retain_size)

        


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget_size", type=int, default=400)
    parser.add_argument("--retain_size", type=int, default=300)
    parser.add_argument("--validate_size", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./preprocessed_data")
    parser.add_argument("--tokenizer", type=str, default="01-ai/Yi-6B")
    
    args = parser.parse_args()
    
    setup = MinimalDataSetup(args.tokenizer)
    setup.setup_minimal_data(
        forget_size=args.forget_size,
        retain_size=args.retain_size,
        validate_size=args.validate_size,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 