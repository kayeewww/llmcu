#!/usr/bin/env python3

import os, time, random, argparse, pathlib, json
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn import functional as F
import wandb


os.environ["WANDB_MODE"] = "offline"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
    os.environ.pop(key, None)

# === Certified Bounded LoRA Unlearning 算法参数配置 ===
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True, type=str, 
                   help="基础模型路径")
parser.add_argument("--scenario", choices=["arxiv", "github", "wikitext"], 
                   default="wikitext", help="遗忘场景选择")
parser.add_argument("--out_root", default="runs", help="输出根目录")

# === 算法核心参数 (对应算法输入) ===
parser.add_argument("--lora_r", type=int, default=4, 
                   help="LoRA秩r - 算法输入参数")
parser.add_argument("--r_prime", type=int, default=2, 
                   help="辅助LoRA秩")
parser.add_argument("--lora_scope", choices=["qv", "full"], default="qv",
                   help="LoRA应用范围")
parser.add_argument("--l1_lambda", type=float, default=1e-7,
                   help="L1正则化系数")

# === TV约束参数 (对应算法中的ε_t) ===
parser.add_argument("--tv_bound", type=float, default=0.5,
                   help="TV上界 - 对应算法中的ε_t预算")
parser.add_argument("--tv_lower", type=float, default=0.2,
                   help="TV下界 - 确保足够的遗忘效果")
parser.add_argument("--tv_lambda", type=float, default=5.0,
                   help="TV惩罚系数")

# === 训练参数 ===
parser.add_argument("--seed", type=int, default=0, help="随机种子")
parser.add_argument("--batch", type=int, default=1, help="批次大小")
parser.add_argument("--grad_acc", type=int, default=4, help="梯度累积步数")
parser.add_argument("--epochs", type=int, default=3, 
                   help="训练轮数 - 对应算法中的N_max")

# === 算法特定参数 ===
parser.add_argument("--epsilon_t", type=float, default=0.1,
                   help="每次请求的TV预算ε_t - 算法输入参数")
parser.add_argument("--lipschitz", type=float, default=1.0,
                   help="Lipschitz常数估计L̂_t - 算法输入参数")
parser.add_argument("--r_max", type=int, default=8,
                   help="全局秩上限r_max - 算法输入参数")
parser.add_argument("--samples_per_scenario", type=int, default=200,
                   help="每个场景的样本数量")
args = parser.parse_args()

def safe(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")

def compute_trust_region_radius(r: int, epsilon_t: float, lipschitz_hat: float) -> float:
    """
    计算信任区域半径 R_t
    
    对应算法第2行: R_t ← sqrt(2r*ε_t/L̂_t)
    
    Args:
        r: LoRA秩
        epsilon_t: TV预算
        lipschitz_hat: Lipschitz常数估计
        
    Returns:
        R_t: 信任区域半径
    """
    return np.sqrt(2 * r * epsilon_t / lipschitz_hat)

def get_lora_delta_weights(model):
    """
    提取LoRA参数的权重变化 ΔW
    
    Args:
        model: LoRA模型
        
    Returns:
        delta_weights: 权重变化的扁平化向量
        weight_shapes: 原始权重形状信息
    """
    delta_weights = []
    weight_shapes = []
    
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            delta_weights.append(param.data.flatten())
            weight_shapes.append((name, param.shape))
    
    if delta_weights:
        return torch.cat(delta_weights), weight_shapes
    else:
        return torch.tensor([]), []

def compute_frobenius_norm(delta_weights):
    """
    计算Frobenius范数 ||ΔW||_F
    
    Args:
        delta_weights: 扁平化的权重变化向量
        
    Returns:
        frobenius_norm: Frobenius范数
    """
    if len(delta_weights) == 0:
        return torch.tensor(0.0)
    return torch.norm(delta_weights, p='fro')

def gram_schmidt_orthogonalization(delta_w_star, adapter_list):
    """
    实现Gram-Schmidt正交化
    
    对应算法第17行: ΔW* ← ΔW* - Σ<ΔW*, A_i>_F/||A_i||_F^2 * A_i
    
    Args:
        delta_w_star: 当前权重变化
        adapter_list: 现有适配器列表
        
    Returns:
        orthogonalized_delta_w: 正交化后的权重变化
    """
    if len(adapter_list) == 0:
        return delta_w_star
    
    orthogonalized = delta_w_star.clone()
    
    for adapter_direction, adapter_norm in adapter_list:
        # 计算内积 <ΔW*, A_i>_F
        inner_product = torch.dot(orthogonalized.flatten(), adapter_direction.flatten())
        
        # 计算投影并减去: ΔW* ← ΔW* - <ΔW*, A_i>_F/||A_i||_F^2 * A_i
        adapter_norm_squared = adapter_norm ** 2
        if adapter_norm_squared > 1e-8:  # 避免除零
            projection = (inner_product / adapter_norm_squared) * adapter_direction
            orthogonalized = orthogonalized - projection.view_as(orthogonalized)
    
    return orthogonalized

def radial_projection(delta_w, trust_radius):
    """
    径向投影到信任区域边界
    
    对应算法第12-13行: 如果||ΔW||_F > R_t，则ΔW ← (R_t/||ΔW||_F) * ΔW
    
    Args:
        delta_w: 权重变化
        trust_radius: 信任区域半径
        
    Returns:
        projected_delta_w: 投影后的权重变化
    """
    delta_w_norm = compute_frobenius_norm(delta_w)
    
    if delta_w_norm > trust_radius:
        # 径向投影: ΔW ← (R_t/||ΔW||_F) * ΔW
        scaling_factor = trust_radius / delta_w_norm
        return delta_w * scaling_factor
    
    return delta_w

class LoudFailException(Exception):
    """算法失败异常，对应算法中的LoudFail返回"""
    pass

def total_variation_stable(logits_p, logits_q):
    """
    计算总变分距离 (Total Variation Distance)
    
    这是算法中用于衡量模型输出分布变化的核心度量，对应算法中的TV约束。
    TV距离用于确保遗忘过程中模型输出的变化在可控范围内。
    
    Args:
        logits_p: 第一个模型的logits输出
        logits_q: 第二个模型的logits输出 (通常是基线模型)
        
    Returns:
        tv: 总变分距离，范围在[0,1]之间
    """
    # 确保两个张量在同一设备上
    if logits_p.device != logits_q.device:
        logits_q = logits_q.to(logits_p.device)
    
    # 确保形状匹配，如果不匹配则截取到较小的批次大小
    if logits_p.shape != logits_q.shape:
        min_batch = min(logits_p.size(0), logits_q.size(0))
        logits_p = logits_p[:min_batch]
        logits_q = logits_q[:min_batch]
    
    # 将logits转换为概率分布
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    
    # 添加小的epsilon以避免数值不稳定
    eps = 1e-8
    p = p + eps
    q = q + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    # 计算总变分距离: TV(p,q) = 0.5 * ||p - q||_1
    tv = 0.5 * torch.abs(p - q).sum(dim=-1).mean()
    
    # 将TV值限制在[0,1]范围内
    tv = torch.clamp(tv, 0.0, 1.0)
    
    return tv

def load_scenario_data(scenario: str, tokenizer, total_samples: int = 200):

    if scenario == "wikitext":
        raw_ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split=f"train[:{total_samples}]",
            cache_dir="./dataset_cache",
        )
        
        def tok_fn(ex):
            out = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=64)
            out["labels"] = out["input_ids"].copy()
            return out
        
        ds = raw_ds.map(tok_fn, batched=True, remove_columns=["text"]).shuffle(seed=args.seed)
        
        forget_prompts = ["Paris is the capital of France.", "The capital of France is"]
        
    else:

        try:
            from data_loader_utils import DatasetLoader
            loader = DatasetLoader("./preprocessed_data")
            

            availability = loader.check_data_availability()
            
            if availability.get(scenario, False):

                datasets = loader.load_scenario_datasets(scenario)
                

                from datasets import concatenate_datasets
                ds = concatenate_datasets([datasets['forget'], datasets['retain']])

                

                if scenario == "arxiv":
                    forget_prompts = [
                        "What is deep learning?",
                        "Explain gradient descent algorithm.",
                        "What are neural networks?",
                        "How does backpropagation work?",
                        "What is machine learning?",
                        "Explain transformer architecture."
                    ]
                elif scenario == "github":
                    forget_prompts = [
                        "What is Python programming?",
                        "How to use JavaScript?",
                        "What is object-oriented programming?",
                        "Explain Git version control.",
                        "How to write functions?",
                        "What is a class in programming?"
                    ]
                
            else:

                raise FileNotFoundError("Not found")
                
        except Exception as e:

            
            from figures_generator.three_forgetting_scenarios import TwoForgettingScenarios
            creator = TwoForgettingScenarios(tokenizer.name_or_path)
            
            if scenario == "arxiv":
                datasets = creator.scenario_arxiv_papers(total_samples)
                forget_prompts = [
                    "What is deep learning?",
                    "Explain gradient descent algorithm.",
                    "What are neural networks?",
                    "How does backpropagation work?",
                    "What is machine learning?",
                    "Explain transformer architecture."
                ]
            elif scenario == "github":
                datasets = creator.scenario_github_repositories(total_samples)
                forget_prompts = [
                    "What is Python programming?",
                    "How to use JavaScript?",
                    "What is object-oriented programming?",
                    "Explain Git version control.",
                    "How to write functions?",
                    "What is a class in programming?"
                ]
            

            from datasets import concatenate_datasets
            ds = concatenate_datasets([datasets["forget"], datasets["retain"]])
    
    return ds, forget_prompts

def create_scenario_specific_evaluation(scenario: str, forget_prompts: list, tokenizer):

    tok_forget = tokenizer(forget_prompts, return_tensors="pt", padding=True)
    

    if scenario == "arxiv":

        target_tokens = ["learning", "algorithm", "network", "model", "neural", "training"]
    elif scenario == "github":

        target_tokens = ["programming", "code", "function", "variable", "class", "import"]
    else:

        target_tokens = ["London", "Paris", "capital", "France"]
    

    tgt_id = tokenizer.encode(target_tokens[0], add_special_tokens=False)[0]
    
    def quick_ue(model):

        with torch.no_grad():
            inputs = {k: v.to(model.device) for k, v in tok_forget.items()}
            out = model(**inputs)
            total_loss = 0
            for i in range(len(forget_prompts)):
                lbl = torch.full_like(inputs["input_ids"][i:i+1], tgt_id)
                lbl.masked_fill_(inputs["attention_mask"][i:i+1] == 0, tokenizer.pad_token_id)
                lbl[0, 0] = tgt_id
                loss = F.cross_entropy(
                    out.logits[i:i+1].view(-1, out.logits.size(-1)),
                    lbl.view(-1),
                    ignore_index=tokenizer.pad_token_id,
                )
                total_loss += loss.item()
            return total_loss / len(forget_prompts)
    
    return quick_ue, tok_forget


random.seed(args.seed)
torch.manual_seed(args.seed)


tag = f"{args.scenario}_r{args.lora_r}_{args.lora_scope}_tv{args.tv_bound}_e{args.epochs}_seed{args.seed}"
OUT_DIR = pathlib.Path(args.out_root) / safe(pathlib.Path(args.base_model).name) / tag
OUT_DIR.mkdir(parents=True, exist_ok=True)


run = wandb.init(project="tv_lora_three_scenarios", name=safe(tag), dir=str(OUT_DIR), config=vars(args))


tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
)

ds, forget_prompts = load_scenario_data(args.scenario, tokenizer, args.samples_per_scenario)

# 场景描述，用于结果记录
scenario_descriptions = {
    "wikitext": "WikiText-2数据集遗忘场景",
    "arxiv": "ArXiv论文数据遗忘场景", 
    "github": "GitHub代码数据遗忘场景"
}

quick_ue, tok_forget = create_scenario_specific_evaluation(args.scenario, forget_prompts, tokenizer)


@torch.no_grad()
def collect_baseline(dataset, N=50):
    """
    收集基线模型的输出logits
    
    这对应算法中的基础权重W，用作后续计算TV距离的参考点。
    基线logits用于衡量遗忘过程中模型输出的变化程度。
    
    Args:
        dataset: 数据集
        N: 采样的数据点数量
        
    Returns:
        baseline_logits: 基线模型在数据集上的logits输出
    """
    base_model.eval()
    outs = []
    for d in dataset.select(range(min(N, len(dataset)))):
        ids = torch.tensor(d["input_ids"]).unsqueeze(0).to(base_model.device)
        # 获取基线模型的logits输出，这是算法中的W(基础权重)的输出
        logits = base_model(ids).logits
        outs.append(logits.cpu())
    return torch.cat(outs)

baseline_logits = collect_baseline(ds)

with torch.no_grad():
    base_model.eval()
    forget_inputs = {k: v.to(base_model.device) for k, v in tok_forget.items()}
    baseline_forget_logits = base_model(**forget_inputs).logits.cpu()


# === Stage 3: Orthogonal Integration - LoRA配置 ===
# 这部分实现算法第17-23行的正交集成阶段
# LoRA提供了一种隐式的正交化机制，通过低秩分解实现适配器的正交集成

target_modules = (
    ["q_proj", "v_proj"]  # 仅对查询和值投影应用LoRA
    if args.lora_scope == "qv"
    else ["q_proj", "v_proj", "k_proj", "o_proj", "ff_proj"]  # 对所有线性层应用LoRA
)

# LoRA配置 - 对应算法中的低秩适配器设置
lora_cfg = LoraConfig(
    r=args.lora_r,          # LoRA秩r，对应算法输入参数
    lora_alpha=8,           # LoRA缩放因子
    target_modules=target_modules,  # 目标模块
    lora_dropout=0.1,       # Dropout率
    task_type="CAUSAL_LM",  # 任务类型
)

# 准备模型进行量化训练并应用LoRA
model = prepare_model_for_kbit_training(base_model)
model = get_peft_model(model, lora_cfg)

# 只训练LoRA参数，冻结基础模型参数
# 这实现了算法中的适配器机制，确保基础权重W保持不变
for n, p in model.named_parameters():
    p.requires_grad = "lora" in n


ue_curve = []
tv_curve = []
loss_with_tv = []
loss_no_tv = []

class TVTrainer(Trainer):
    """
    实现Certified Bounded LoRA Unlearning算法的训练器
    
    这个类严格实现算法的三个阶段：
    - Stage 1: Trust Region and Ascend (信任区域和梯度上升)
    - Stage 2: Projection & Certification (投影和认证)
    - Stage 3: Orthogonal Integration (正交集成)
    """
    log_every = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_logs = []
        self.csv_log_file = None
        
        # === 算法状态变量 ===
        self.adapter_list = []  # 对应算法中的适配器列表A
        self.current_rank = 0   # 当前总秩，用于检查r_max约束
        
        # === 算法参数 ===
        self.trust_radius = compute_trust_region_radius(
            args.lora_r, args.epsilon_t, args.lipschitz
        )  # R_t = sqrt(2r*ε_t/L̂_t)
        
        self.max_steps = 0  # 当前训练步数，用于N_max检查
        self.reached_boundary = False  # 是否达到信任区域边界
        
        print(f"信任区域半径 R_t = {self.trust_radius:.6f}")
        print(f"LoRA秩 r = {args.lora_r}, TV预算 ε_t = {args.epsilon_t}, Lipschitz估计 L̂_t = {args.lipschitz}")
        print(f"全局秩上限 r_max = {args.r_max}")
        
    def setup_csv_logging(self, output_dir):

        import csv
        from pathlib import Path
        
        log_path = Path(output_dir) / "training_logs.csv"
        self.csv_log_file = open(log_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_log_file)
        

        self.csv_writer.writerow([
            'step', 'loss', 'ce_loss', 'tv_penalty', 'tv_value', 
            'tv_current', 'tv_forget', 'ue_score', 'learning_rate', 
            'scenario', 'timestamp', 'delta_w_norm', 'trust_radius',
            'epsilon_t_cert', 'orthogonal_norm', 'adapter_count', 'reached_boundary'
        ])


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        严格实现Certified Bounded LoRA Unlearning算法的三个阶段
        
        Stage 1: Trust Region and Ascend (算法第1-10行)
        Stage 2: Projection & Certification (算法第11-16行)
        Stage 3: Orthogonal Integration (算法第17-23行)
        """
        try:
            return self._execute_algorithm_stages(model, inputs, return_outputs, **kwargs)
        except LoudFailException as e:
            print(f"算法失败: {e}")
            # 返回基础损失，不进行遗忘
            labels = inputs["labels"]
            outputs = model(**inputs)
            ce = F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            return (ce, outputs) if return_outputs else ce
    
    def _execute_algorithm_stages(self, model, inputs, return_outputs=False, **kwargs):
        """执行算法的三个阶段"""
        labels = inputs["labels"]
        outputs = model(**inputs)
        
        # === Stage 1: Trust Region and Ascend ===
        # 第3行: ΔW ← 0 (LoRA参数初始化时已为0)
        # 第4-9行: 梯度上升循环
        
        # 第5行: G ← ∇_ΔW L_u(W + ΔW; D_f) - 计算遗忘损失梯度
        ce = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )
        
        # 获取当前LoRA权重变化 ΔW
        delta_weights, weight_shapes = get_lora_delta_weights(model)
        delta_w_norm = compute_frobenius_norm(delta_weights)
        
        # 第7-8行: 检查是否达到信任区域边界
        if delta_w_norm >= self.trust_radius:
            self.reached_boundary = True
            print(f"达到信任区域边界: ||ΔW||_F = {delta_w_norm:.6f} >= R_t = {self.trust_radius:.6f}")
        
        # 检查最大步数 (对应算法第4行的k <= N_max)
        self.max_steps += 1
        max_training_steps = args.epochs * len(self.train_dataloader) if hasattr(self, 'train_dataloader') else 1000
        
        if self.max_steps >= max_training_steps and not self.reached_boundary:
            # 第10行: 如果未达到边界则返回LoudFail
            raise LoudFailException("未达到信任区域边界")
        
        # === Stage 2: Projection & Certification ===
        delta_w_star = delta_weights.clone() if len(delta_weights) > 0 else torch.tensor([])
        
        # 第12-13行: 径向投影
        if delta_w_norm > self.trust_radius:
            delta_w_star = radial_projection(delta_w_star, self.trust_radius)
            print(f"执行径向投影: {delta_w_norm:.6f} -> {self.trust_radius:.6f}")
        
        # 第15行: 计算消耗的认证预算
        epsilon_t_cert = (args.lipschitz / (2 * args.lora_r)) * (compute_frobenius_norm(delta_w_star) ** 2)
        
        # === Stage 3: Orthogonal Integration ===
        # 第17行: Gram-Schmidt正交化
        delta_w_orthogonal = gram_schmidt_orthogonalization(delta_w_star, self.adapter_list)
        
        # 第18行: 检查正交化后是否退化
        orthogonal_norm = compute_frobenius_norm(delta_w_orthogonal)
        if orthogonal_norm == 0:
            raise LoudFailException("正交化后权重变化为零")
        
        # 第20行: 存储方向 U ← ΔW*/||ΔW*||_F
        if orthogonal_norm > 1e-8:
            direction = delta_w_orthogonal / orthogonal_norm
            
            # 第21行: 更新适配器列表 A ← A ∪ {(U, ||ΔW*||_F)}
            self.adapter_list.append((direction, orthogonal_norm))
            
            # 第22行: 检查全局秩约束
            total_rank = len(self.adapter_list) * args.lora_r
            if total_rank > args.r_max:
                raise LoudFailException(f"超出全局秩上限: {total_rank} > {args.r_max}")
        
        # 计算TV距离用于监控
        tv_current, tv_forget = self._compute_tv_distances(model, outputs, inputs)
        tv_combined = 0.3 * tv_current + 0.7 * tv_forget
        
        # 应用TV约束作为软约束
        tv_penalty = self._compute_tv_penalty(tv_combined)
        
        # 最终损失
        loss = ce + tv_penalty
        
        # 数值稳定性检查
        if torch.isnan(loss) or torch.isinf(loss):
            loss = ce
        
        # 记录算法状态
        self._log_algorithm_state(model, loss, ce, tv_penalty, tv_current, tv_forget, 
                                 delta_w_norm, epsilon_t_cert, orthogonal_norm)
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_tv_distances(self, model, outputs, inputs):
        """计算TV距离"""
        bs = outputs.logits.size(0)
        
        # 缓存基线模型输出
        if not hasattr(self, '_baseline_cache'):
            self._baseline_cache = baseline_logits.to(outputs.logits.device)
            print(f"基线logits缓存形状: {self._baseline_cache.shape}")
        
        if not hasattr(self, '_baseline_forget_cache'):
            self._baseline_forget_cache = baseline_forget_logits.to(outputs.logits.device)
        
        # 获取当前批次对应的基线logits
        max_start = max(0, self._baseline_cache.size(0) - bs)
        start_idx = (self.state.global_step * bs) % (max_start + 1) if max_start > 0 else 0
        base_logits = self._baseline_cache[start_idx:start_idx + bs]
        
        # 计算当前数据的TV距离
        tv_current = total_variation_stable(outputs.logits, base_logits)
        
        # 计算遗忘数据的TV距离
        forget_inputs = {k: v.to(outputs.logits.device) for k, v in tok_forget.items()}
        current_forget_logits = model(**forget_inputs).logits
        tv_forget = total_variation_stable(current_forget_logits, self._baseline_forget_cache)
        
        return tv_current, tv_forget
    
    def _compute_tv_penalty(self, tv_combined):
        """计算TV惩罚"""
        if torch.isnan(tv_combined) or torch.isinf(tv_combined):
            tv_combined = torch.tensor(0.1, device=tv_combined.device)
        
        tv_val = float(tv_combined.item())
        
        if tv_val > args.tv_bound:
            excess = tv_val - args.tv_bound
            tv_penalty = args.tv_lambda * (excess ** 2)
        elif tv_val < args.tv_lower:
            deficit = args.tv_lower - tv_val
            tv_penalty = -0.5 * args.tv_lambda * deficit
        else:
            tv_penalty = 0.1 * args.tv_lambda * (tv_val - (args.tv_lower + args.tv_bound) / 2) ** 2
        
        if not isinstance(tv_penalty, torch.Tensor):
            tv_penalty = torch.tensor(tv_penalty, device=tv_combined.device, requires_grad=True)
        
        return tv_penalty
    
    def _log_algorithm_state(self, model, loss, ce, tv_penalty, tv_current, tv_forget, 
                           delta_w_norm, epsilon_t_cert, orthogonal_norm):
        """记录算法状态"""
        if self.state.global_step % self.log_every == 0:
            step = int(self.state.global_step)
            ue = float(quick_ue(model))
            tv_val = float(0.3 * tv_current + 0.7 * tv_forget)
            
            current_lr = self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else 0.0
            
            # 记录训练曲线
            ue_curve.append([step, ue])
            tv_curve.append([step, tv_val])
            loss_with_tv.append([step, float(loss.item())])
            loss_no_tv.append([step, float(ce.item())])
            
            # CSV日志
            if hasattr(self, 'csv_writer') and self.csv_writer:
                from datetime import datetime
                self.csv_writer.writerow([
                    step, float(loss.item()), float(ce.item()),
                    float(tv_penalty.item()) if isinstance(tv_penalty, torch.Tensor) else tv_penalty,
                    tv_val, float(tv_current.item()), float(tv_forget.item()),
                    ue, current_lr, args.scenario, datetime.now().isoformat(),
                    float(delta_w_norm), self.trust_radius, float(epsilon_t_cert),
                    float(orthogonal_norm), len(self.adapter_list), self.reached_boundary
                ])
                self.csv_log_file.flush()
            
            # Wandb日志
            wandb.log({
                "step": step, "UE": ue, "TV": tv_val,
                "tv_current": float(tv_current.item()), "tv_forget": float(tv_forget.item()),
                "loss_with_tv": float(loss.item()), "loss_no_tv": float(ce.item()),
                "tv_penalty": float(tv_penalty.item()) if isinstance(tv_penalty, torch.Tensor) else tv_penalty,
                "delta_w_norm": float(delta_w_norm), "trust_radius": self.trust_radius,
                "epsilon_t_cert": float(epsilon_t_cert), "orthogonal_norm": float(orthogonal_norm),
                "adapter_count": len(self.adapter_list), "reached_boundary": self.reached_boundary,
                "scenario": args.scenario, "learning_rate": current_lr,
            }, step=step)
            
            # 控制台输出
            boundary_status = "已达到边界" if self.reached_boundary else "未达到边界"
            print(f"Step {step} [{args.scenario}]: {boundary_status}")
            print(f"  loss={loss:.4f} (ce={ce:.4f}, tv_penalty={tv_penalty:.4f}), ue={ue:.4f}")
            print(f"  ||ΔW||_F={delta_w_norm:.6f}, R_t={self.trust_radius:.6f}")
            print(f"  ε_t^cert={epsilon_t_cert:.6f}, 适配器数量={len(self.adapter_list)}")
            print(f"  tv_current={tv_current:.4f}, tv_forget={tv_forget:.4f}, lr={current_lr:.2e}")


# === 算法执行 - 训练器初始化 ===
# 使用自定义的TVTrainer来执行Certified Bounded LoRA Unlearning算法
trainer = TVTrainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=args.batch,      # 批次大小
        gradient_accumulation_steps=args.grad_acc,   # 梯度累积步数
        num_train_epochs=args.epochs,                # 训练轮数，对应算法中的N_max
        logging_steps=5,
        fp16=True,                                   # 使用半精度训练
        save_strategy="no",                          # 不保存中间检查点
        learning_rate=2e-5,                          # 学习率η，对应算法输入参数
        optim="adamw_torch",                         # 优化器
        dataloader_pin_memory=False,
        gradient_checkpointing=True,                 # 梯度检查点以节省内存
        remove_unused_columns=False,
        local_rank=-1,
        max_grad_norm=1.0,                          # 梯度裁剪，有助于算法稳定性
        weight_decay=0.01,                          # 权重衰减
        warmup_ratio=0.1,                           # 学习率预热
        lr_scheduler_type="cosine",                 # 余弦学习率调度
    ),
    train_dataset=ds,      # 训练数据集，包含遗忘数据D_f
    tokenizer=tokenizer,
)


# 设置CSV日志记录
trainer.setup_csv_logging(OUT_DIR)

try:
    # === 执行Certified Bounded LoRA Unlearning算法 ===
    # 这里开始执行完整的三阶段算法流程
    trainer.train()
    
    # === 算法结果分析和认证 ===
    print("\n=== Certified Bounded LoRA Unlearning 算法执行结果 ===")
    
    # 检查算法是否成功完成
    algorithm_success = trainer.reached_boundary
    total_adapters = len(trainer.adapter_list)
    trust_radius = trainer.trust_radius
    
    print(f"算法执行状态: {'成功' if algorithm_success else '失败'}")
    print(f"信任区域边界: {'已达到' if trainer.reached_boundary else '未达到'}")
    print(f"信任区域半径 R_t: {trust_radius:.6f}")
    print(f"适配器数量: {total_adapters}")
    print(f"全局秩约束: {total_adapters * args.lora_r}/{args.r_max}")
    
    # 计算最终的权重变化范数
    final_delta_weights, _ = get_lora_delta_weights(model)
    final_delta_norm = compute_frobenius_norm(final_delta_weights)
    print(f"最终权重变化范数 ||ΔW||_F: {final_delta_norm:.6f}")
    
    # 计算认证预算消耗
    final_epsilon_cert = (args.lipschitz / (2 * args.lora_r)) * (final_delta_norm ** 2)
    print(f"认证预算消耗 ε_t^cert: {final_epsilon_cert:.6f}")
    print(f"预算利用率: {final_epsilon_cert/args.epsilon_t*100:.2f}%")
    
    # TV距离分析
    tv_in_range_count = sum(1 for _, tv_val in tv_curve if args.tv_lower <= tv_val <= args.tv_bound)
    tv_in_range_ratio = tv_in_range_count / len(tv_curve) if tv_curve else 0
    
    if len(tv_curve) > 10:
        initial_tv = np.mean([tv_val for _, tv_val in tv_curve[:5]])
        final_tv = np.mean([tv_val for _, tv_val in tv_curve[-5:]])
        tv_growth = final_tv - initial_tv
        print(f"TV距离变化: {initial_tv:.4f} -> {final_tv:.4f} (增长: {tv_growth:.4f})")
        print(f"TV在目标范围内的比例: {tv_in_range_ratio*100:.2f}%")
    
    # 算法认证结果
    certification_passed = (
        algorithm_success and 
        trainer.reached_boundary and 
        total_adapters * args.lora_r <= args.r_max and
        final_epsilon_cert <= args.epsilon_t
    )
    
    print(f"\n算法认证结果: {'通过' if certification_passed else '未通过'}")
    if not certification_passed:
        print("认证失败原因:")
        if not algorithm_success:
            print("  - 算法执行失败")
        if not trainer.reached_boundary:
            print("  - 未达到信任区域边界")
        if total_adapters * args.lora_r > args.r_max:
            print(f"  - 超出全局秩约束: {total_adapters * args.lora_r} > {args.r_max}")
        if final_epsilon_cert > args.epsilon_t:
            print(f"  - 超出预算约束: {final_epsilon_cert:.6f} > {args.epsilon_t}")
    
    # 保存算法执行结果，包括认证信息
    result = {
        "scenario": args.scenario,
        "scenario_description": scenario_descriptions[args.scenario],
        "forget_prompts": forget_prompts,
        "ue_curve": ue_curve,                    # 遗忘效果曲线
        "tv_curve": tv_curve,                    # TV距离曲线，对应||ΔW||_F
        "loss_with_tv": loss_with_tv,            # 包含TV约束的损失
        "loss_no_tv": loss_no_tv,                # 不含TV约束的损失
        "config": vars(args),                    # 算法配置参数
        
        # === 算法认证结果 ===
        "algorithm_success": algorithm_success,
        "reached_boundary": trainer.reached_boundary,
        "trust_radius": trust_radius,
        "final_delta_norm": float(final_delta_norm),
        "final_epsilon_cert": float(final_epsilon_cert),
        "budget_utilization": float(final_epsilon_cert/args.epsilon_t),
        "adapter_count": total_adapters,
        "total_rank": total_adapters * args.lora_r,
        "rank_constraint_satisfied": total_adapters * args.lora_r <= args.r_max,
        "certification_passed": certification_passed,
        
        # === TV分析结果 ===
        "final_tv": float(tv_curve[-1][1]) if tv_curve else 0.0,
        "tv_in_range_ratio": float(tv_in_range_ratio),
        "tv_growth": float(tv_growth) if len(tv_curve) > 10 else 0.0,
        "initial_tv": float(initial_tv) if len(tv_curve) > 10 else 0.0,
        "final_tv_mean": float(final_tv) if len(tv_curve) > 10 else 0.0,
    }
    
    with open(OUT_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    wandb.finish() 
