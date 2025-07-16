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

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True, type=str)
parser.add_argument("--scenario", choices=["arxiv", "github", "wikitext"], 
                   default="wikitext")
parser.add_argument("--out_root", default="runs")
parser.add_argument("--lora_r", type=int, default=4)
parser.add_argument("--r_prime", type=int, default=2)
parser.add_argument("--lora_scope", choices=["qv", "full"], default="qv")
parser.add_argument("--l1_lambda", type=float, default=1e-7)
parser.add_argument("--tv_bound", type=float, default=0.5)
parser.add_argument("--tv_lower", type=float, default=0.2)
parser.add_argument("--tv_lambda", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--grad_acc", type=int, default=4)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--epsilon_t", type=float, default=0.1)
parser.add_argument("--lipschitz", type=float, default=1.0)
parser.add_argument("--r_max", type=int, default=8)
parser.add_argument("--samples_per_scenario", type=int, default=200)
args = parser.parse_args()

def safe(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")

def total_variation_stable(logits_p, logits_q):

    if logits_p.device != logits_q.device:
        logits_q = logits_q.to(logits_p.device)
    

    if logits_p.shape != logits_q.shape:
        min_batch = min(logits_p.size(0), logits_q.size(0))
        logits_p = logits_p[:min_batch]
        logits_q = logits_q[:min_batch]
    

    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    

    eps = 1e-8
    p = p + eps
    q = q + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    

    tv = 0.5 * torch.abs(p - q).sum(dim=-1).mean()
    

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


quick_ue, tok_forget = create_scenario_specific_evaluation(args.scenario, forget_prompts, tokenizer)


@torch.no_grad()
def collect_baseline(dataset, N=50):

    base_model.eval()
    outs = []
    for d in dataset.select(range(min(N, len(dataset)))):
        ids = torch.tensor(d["input_ids"]).unsqueeze(0).to(base_model.device)

        logits = base_model(ids).logits
        outs.append(logits.cpu())
    return torch.cat(outs)

baseline_logits = collect_baseline(ds)

with torch.no_grad():
    base_model.eval()
    forget_inputs = {k: v.to(base_model.device) for k, v in tok_forget.items()}
    baseline_forget_logits = base_model(**forget_inputs).logits.cpu()


target_modules = (
    ["q_proj", "v_proj"]
    if args.lora_scope == "qv"
    else ["q_proj", "v_proj", "k_proj", "o_proj", "ff_proj"]
)

lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=8,
    target_modules=target_modules,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(base_model)
model = get_peft_model(model, lora_cfg)


for n, p in model.named_parameters():
    p.requires_grad = "lora" in n


ue_curve = []
tv_curve = []
loss_with_tv = []
loss_no_tv = []

class TVTrainer(Trainer):
    log_every = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_logs = []
        self.csv_log_file = None
        
    def setup_csv_logging(self, output_dir):

        import csv
        from pathlib import Path
        
        log_path = Path(output_dir) / "training_logs.csv"
        self.csv_log_file = open(log_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_log_file)
        

        self.csv_writer.writerow([
            'step', 'loss', 'ce_loss', 'tv_penalty', 'tv_value', 
            'tv_current', 'tv_forget', 'ue_score', 'learning_rate', 
            'scenario', 'timestamp'
        ])


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)


        ce = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )


        bs = outputs.logits.size(0)
        

        if not hasattr(self, '_baseline_cache'):
            self._baseline_cache = baseline_logits.to(outputs.logits.device)
            print(f"{self._baseline_cache.shape}")
        
        if not hasattr(self, '_baseline_forget_cache'):
            self._baseline_forget_cache = baseline_forget_logits.to(outputs.logits.device)
        

        max_start = max(0, self._baseline_cache.size(0) - bs)
        start_idx = (self.state.global_step * bs) % (max_start + 1) if max_start > 0 else 0
        base_logits = self._baseline_cache[start_idx:start_idx + bs]
        

        tv_current = total_variation_stable(outputs.logits, base_logits)
        

        forget_inputs = {k: v.to(outputs.logits.device) for k, v in tok_forget.items()}
        current_forget_logits = model(**forget_inputs).logits
        

        tv_forget = total_variation_stable(current_forget_logits, self._baseline_forget_cache)
        

        tv = 0.3 * tv_current + 0.7 * tv_forget
        

        if torch.isnan(tv) or torch.isinf(tv):

            tv = torch.tensor(0.1, device=tv.device)
        

        tv_val = float(tv.item())
        
        if tv_val > args.tv_bound:

            excess = tv_val - args.tv_bound
            tv_penalty = args.tv_lambda * (excess ** 2)
        elif tv_val < args.tv_lower:

            deficit = args.tv_lower - tv_val
            tv_penalty = -0.5 * args.tv_lambda * deficit
        else:

            tv_penalty = 0.1 * args.tv_lambda * (tv_val - (args.tv_lower + args.tv_bound) / 2) ** 2
        

        if not isinstance(tv_penalty, torch.Tensor):
            tv_penalty = torch.tensor(tv_penalty, device=tv.device, requires_grad=True)
            
        loss = ce + tv_penalty
        

        if torch.isnan(loss) or torch.isinf(loss):
            loss = ce


        if self.state.global_step % self.log_every == 0:
            step = int(self.state.global_step)
            ue = float(quick_ue(model))
            in_range = args.tv_lower <= tv_val <= args.tv_bound
            

            current_lr = self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else 0.0


            ue_curve.append([step, ue])
            tv_curve.append([step, tv_val])
            loss_with_tv.append([step, float(loss.item())])
            loss_no_tv.append([step, float(ce.item())])


            if hasattr(self, 'csv_writer') and self.csv_writer:
                from datetime import datetime
                self.csv_writer.writerow([
                    step,
                    float(loss.item()),
                    float(ce.item()),
                    float(tv_penalty.item()) if isinstance(tv_penalty, torch.Tensor) else tv_penalty,
                    tv_val,
                    float(tv_current.item()),
                    float(tv_forget.item()),
                    ue,
                    current_lr,
                    args.scenario,
                    datetime.now().isoformat()
                ])
                self.csv_log_file.flush()  


            wandb.log({
                "step": step,
                "UE": ue,
                "TV": tv_val,
                "tv_current": float(tv_current.item()),
                "tv_forget": float(tv_forget.item()),
                "loss_with_tv": float(loss.item()),
                "loss_no_tv": float(ce.item()),
                "tv_penalty": float(tv_penalty.item()) if isinstance(tv_penalty, torch.Tensor) else tv_penalty,
                "tv_in_range": float(in_range),
                "scenario": args.scenario,
                "learning_rate": current_lr,
            }, step=step)

                
            print(f"Step {step} [{args.scenario}]: {status}")
            print(f"  loss={loss:.4f} (ce={ce:.4f}, tv_penalty={tv_penalty:.4f}), ue={ue:.4f}")
            print(f"  tv_current={tv_current:.4f}, tv_forget={tv_forget:.4f}, lr={current_lr:.2e}")

        return (loss, outputs) if return_outputs else loss


trainer = TVTrainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        logging_steps=5,
        fp16=True,
        save_strategy="no",
        learning_rate=2e-5,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        local_rank=-1,
        max_grad_norm=1.0,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    ),
    train_dataset=ds,
    tokenizer=tokenizer,
)


trainer.setup_csv_logging(OUT_DIR)

try:
    trainer.train()
    
    tv_in_range_count = sum(1 for _, tv_val in tv_curve if args.tv_lower <= tv_val <= args.tv_bound)
    tv_in_range_ratio = tv_in_range_count / len(tv_curve) if tv_curve else 0
    

    if len(tv_curve) > 10:
        initial_tv = np.mean([tv_val for _, tv_val in tv_curve[:5]])
        final_tv = np.mean([tv_val for _, tv_val in tv_curve[-5:]])
        tv_growth = final_tv - initial_tv
        print(f"TV {initial_tv:.4f} -> {final_tv:.4f} ( {tv_growth:.4f})")
    

    result = {
        "scenario": args.scenario,
        "scenario_description": scenario_descriptions[args.scenario],
        "forget_prompts": forget_prompts,
        "ue_curve": ue_curve,
        "tv_curve": tv_curve,
        "loss_with_tv": loss_with_tv,
        "loss_no_tv": loss_no_tv,
        "config": vars(args),
        "final_tv": float(tv_curve[-1][1]) if tv_curve else 0.0,
        "tv_in_range_ratio": float(tv_in_range_ratio),
        "tv_growth": float(tv_growth) if len(tv_curve) > 10 else 0.0,
    }
    
    with open(OUT_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    wandb.finish() 