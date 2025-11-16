# d:\Code\LLM-workspace\MLLM-SFT\args_config.py
import argparse

def _str2bool(v):
    return str(v).lower() in ("1", "true", "t", "yes", "y")

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3-1.7B")
    p.add_argument("--cache-dir", default="./")
    p.add_argument("--revision", default="master")
    p.add_argument("--prompt", default="你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。")
    p.add_argument("--data-max-length", type=int, default=2048)
    p.add_argument("--train-dataset-path", default="train.jsonl")
    p.add_argument("--val-dataset-path", default="val.jsonl")
    p.add_argument("--train-formatted-path", default="train_format.jsonl")
    p.add_argument("--val-formatted-path", default="val_format.jsonl")
    p.add_argument("--output-dir", default="./output/Qwen3-1.7B")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--eval-strategy", default="steps")
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--num-train-epochs", type=int, default=2)
    p.add_argument("--save-steps", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--save-on-each-node", type=_str2bool, default=True)
    p.add_argument("--gradient-checkpointing", type=_str2bool, default=True)
    p.add_argument("--report-to", default="swanlab")
    p.add_argument("--run-name", default="qwen3-1.7B")
    p.add_argument("--swanlab-project", default="qwen3-sft-medical")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--use-fast-tokenizer", type=_str2bool, default=False)
    p.add_argument("--trust-remote-code", type=_str2bool, default=True)
    p.add_argument("--preview-samples-count", type=int, default=3)
    return p