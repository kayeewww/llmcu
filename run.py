#!/usr/bin/env python3

import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="full process")
    parser.add_argument("--model_path", type=str, default="01-ai/Yi-6B", help="basic path")
    parser.add_argument("--scenarios", type=str, nargs="+", 
                       choices=["arxiv", "github", "both"],
                       default=["both"], help="running scenario")
    parser.add_argument("--output_dir", type=str, default=None, help="output catalogue")
    parser.add_argument("--skip_data_setup", action="store_true", help="skip data preprocssing")
    parser.add_argument("--skip_training", action="store_true", help="skip training, only evaluation")
    parser.add_argument("--skip_evaluation", action="store_true", help="skip evaluation, only training")
    parser.add_argument("--quick_test", action="store_true", help="quick test")
    
    args = parser.parse_args()
    

    if "both" in args.scenarios:
        scenarios = ["arxiv", "github"]
    else:
        scenarios = args.scenarios
    

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"complete_experiment_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_data_setup:
        cmd = [
            "python", "data_setup.py",
            "--forget_size", "400",
            "--retain_size", "300", 
            "--validate_size", "200"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Done")
        except subprocess.CalledProcessError as e:
            print(f" {e.stderr}")
            return
    
    if not args.skip_training:
        
        for scenario in scenarios:
            
            scenario_output = output_dir / f"training_{scenario}"
            scenario_output.mkdir(exist_ok=True)
            
            cmd = [
                "python", "unlearning.py",
                "--base_model", args.model_path,
                "--scenario", scenario,
                "--out_root", str(scenario_output),
                "--epochs", "1", 
                "--batch", "2",
                "--grad_acc", "2",
                "--lora_r", "4",
                "--tv_lambda", "5.0",
                "--tv_bound", "0.5",
                "--tv_lower", "0.2"
            ]
            
            if args.quick_test:
                cmd.extend(["--samples_per_scenario", "50"])
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("complete")
            except subprocess.CalledProcessError as e:
                print(f"{e.stderr}")
                continue
    
    if not args.skip_evaluation:
        
        for scenario in scenarios:
            
            eval_output = output_dir / f"evaluation_{scenario}"
            eval_output.mkdir(exist_ok=True)
            
            # 查找训练好的模型
            training_dir = output_dir / f"training_{scenario}"
            model_path = None
            
            if training_dir.exists():
                # 查找LoRA模型
                for model_dir in training_dir.rglob("*"):
                    if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
                        model_path = str(model_dir)
                        break
            
            # 运行优化的评估
            cmd = [
                "python", "evaluation.py",
                "--scenario", scenario,
                "--output_dir", str(eval_output),
                "--model_path", args.model_path,
                "--max_samples", "50" if args.quick_test else "100"
            ]
            
            if model_path:
                cmd.extend(["--experiments_dir", str(training_dir)])
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"complete")
            except subprocess.CalledProcessError as e:
                print(f"{e.stderr}")
                continue
            
            
            downstream_cmd = [
                "python", "downstream_evaluator.py",
                "--base_model", args.model_path,
                "--max_samples", "100" if args.quick_test else "500",
                "--output_dir", str(eval_output),
                "--tasks", "all"
            ]
            
            if model_path:
                downstream_cmd.extend(["--model_path", model_path])
            
            try:
                result = subprocess.run(downstream_cmd, check=True, capture_output=True, text=True)
                print(f"complete")
            except subprocess.CalledProcessError as e:
                print(f" {e}")
    
    report_data = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model_path,
            "scenarios": scenarios,
            "output_dir": str(output_dir),
            "quick_test": args.quick_test
        },
        "results": {}
    }
    
    for scenario in scenarios:
        scenario_results = {}
        
        training_dir = output_dir / f"training_{scenario}"
        if training_dir.exists():
            result_file = training_dir / "Yi-6B" / f"{scenario}_r4_qv_tv0.5_e1_seed0" / "result.json"
            if result_file.exists():
                import json
                with open(result_file, 'r') as f:
                    scenario_results["training"] = json.load(f)
        
        eval_dir = output_dir / f"evaluation_{scenario}"
        if eval_dir.exists():
            summary_file = eval_dir / "evaluation_summary.csv"
            if summary_file.exists():
                scenario_results["evaluation"] = str(summary_file)
            
            downstream_files = list(eval_dir.glob("downstream_results_*.json"))
            if downstream_files:
                scenario_results["downstream"] = str(downstream_files[0])
        
        report_data["results"][scenario] = scenario_results
    

    report_file = output_dir / "experiment_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
if __name__ == "__main__":
    main() 