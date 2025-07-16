# Machine Unlearning Complete Experiment Flow

## 📋 Project Overview

This project implements a TV-regularized machine unlearning method with comprehensive experimental evaluation on the Yi-6B model. The experiments include arXiv academic papers and GitHub code repositories as forgetting scenarios, comparing the effectiveness of multiple baseline methods.

## 🏗️ Project Structure

```
.
├── run.py             # Complete experiment flow controller
├── data_setup.py                  # Optimized data preprocessing
├── unlearnin.py              # Core training script
├── evaluation.py  # Comprehensive evaluation system
└──  downstream_evaluator.py      # Downstream task evaluation
```

## 🚀 Quick Start

### Environment Requirements

```bash
# Python 3.11
# CUDA 11.8+

# Install dependencies
pip install torch transformers datasets peft wandb scikit-learn matplotlib seaborn pandas numpy
```

###  Execution

```bash
# 1. Data preprocessing
python data_setup.py

# 2. Training (1 epoch by default)
python unlearning.py --base_model 01-ai/Yi-6B --scenario arxiv --epochs 1

# 3. Evaluation
python evaluation.py --scenario arxiv --max_samples 50

# 4. Downstream tasks
python downstream_evaluator.py --max_samples 100 --tasks all
```

## 📊 Experimental Design

### Datasets

- **arXiv Academic Papers**: From `armanc/scientific_papers`, containing machine learning related papers
- **GitHub Code**: From `bigcode/the-stack`, containing code in multiple programming languages
- **Data Scale**: 400 forget samples, 300 retain samples, 200 validation samples per scenario

### Baseline Methods

1. **Gradient Ascent (GA)**: Perform gradient ascent on forget data
2. **Gradient Difference (GradDiff)**: Compute gradient difference between forget and retain data
3. **Negative Preference Optimization (NPO)**: Preference optimization method
4. Differential Privacy (DP): Add DP into LoRA with fixed bound 1.

### Evaluation Metrics

- **Forgetting Effectiveness**: NLL, PPL, ACC (forget set)
- **Knowledge Retention**: NLL, PPL, ACC (retain set)
- **Privacy Protection**: Membership Inference Attack (MIA)
- **Downstream Performance**: MMLU, GSM8K, ARC, HumanEval
- **Computational Efficiency**: FLOPS

## 🔧 Detailed Usage

### 1. Data Preprocessing

```bash
python setup.py \
    --forget_size 400 \
    --retain_size 300 \
    --validate_size 200 \
    --output_dir ./preprocessed_data
```

**Features**:
- Prioritize local cache to avoid repeated downloads
- Automatically process arXiv and GitHub data
- Create standardized dataset format
- Support backup data generation

### 2. Training Experiment

```bash
python unlearning.py \
    --base_model 01-ai/Yi-6B \
    --scenario arxiv \
    --out_root ./runs \
    --epochs 1 \
    --batch 2 \
    --grad_acc 2 \
    --lora_r 4 \
    --tv_lambda 5.0 \
    --tv_bound 0.5 \
    --tv_lower 0.2
```

**Parameter Description**:
- `--scenario`: Choose scenario (arxiv/github/wikitext)
- `--lora_r`: LoRA rank, controls parameter update amount
- `--tv_lambda`: TV regularization weight
- `--tv_bound`: TV distance upper bound
- `--epochs`: Training epochs (recommend 1 epoch)

### 3. Comprehensive Evaluation

```bash
python evaluation.py \
    --scenario arxiv \
    --output_dir ./evaluation_results \
    --model_path 01-ai/Yi-6B \
    --max_samples 100 \
    --experiments_dir ./runs
```

**Evaluation Content**:
- NLL, PPL, ACC calculation
- Membership Inference Attack
- FLOPS calculation
- Training log analysis

### 4. Downstream Task Evaluation

```bash
python evaluator.py \
    --base_model 01-ai/Yi-6B \
    --max_samples 1000 \
    --output_dir ./downstream_results \
    --tasks all
```

**Supported Tasks**:
- MMLU: Massive Multitask Language Understanding
- GSM8K: Grade School Math 8K
- ARC: AI2 Reasoning Challenge
- HumanEval: Code generation evaluation

## 📈 Result Interpretation

### Key Metrics

1. **Forgetting Effectiveness (Forget Set)**
   - NLL: Negative log-likelihood, higher indicates better forgetting
   - PPL: Perplexity, higher indicates model uncertainty about forgotten content
   - ACC: Accuracy, lower indicates better forgetting effect
   - MIA: Membership inference attack success rate, lower indicates better privacy protection

2. **Knowledge Retention (Retain Set)**
   - NLL/PPL: Should maintain low values
   - ACC: Should maintain high values

3. **Downstream Performance**
   - MMLU: Multitask language understanding capability
   - GSM8K: Mathematical reasoning capability
   - ARC: Abstract reasoning capability
   - HumanEval: Code generation capability

## 🔍 Experimental Configuration

### Recommended Configuration

```python
# Training configuration
training_config = {
    "base_model": "01-ai/Yi-6B",
    "lora_r": 4,
    "tv_lambda": 5.0,
    "tv_bound": 0.5,
    "epochs": 1,
    "batch_size": 2,
    "gradient_accumulation": 2,
    "learning_rate": 2e-4
}

# Evaluation configuration
evaluation_config = {
    "max_samples": 100,
    "max_length": 256,
    "tasks": ["mmlu", "gsm8k", "arc", "humaneval"]
}
```

### Hyperparameter Tuning

1. **TV Weight (tv_lambda)**
   - Range: 1.0-10.0
   - Recommended: 2.0-5.0
   - Impact: Controls privacy protection strength

2. **LoRA Rank (lora_r)**
   - Range: 2-16
   - Recommended: 4
   - Impact: Parameter update amount and computational efficiency

3. **Training Epochs (epochs)**
   - Recommended: 1 epoch
   - Reason: Avoid overfitting, maintain efficiency
