import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from args_config import build_parser

args = build_parser().parse_args()
os.environ["SWANLAB_PROJECT"] = args.swanlab_project
PROMPT = args.prompt
MAX_LENGTH = args.data_max_length

swanlab.config.update({
    "model": args.model_path,
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
})

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input = data["question"]
            think = data["think"]
            answer = data["answer"]
            output = f"<think>{think}</think> \n {answer}"
            message = {
                "instruction": PROMPT,
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """
    将数据集进行预处理
    """ 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

model_dir = snapshot_download(args.model_path, cache_dir=args.cache_dir, revision=args.revision)

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=args.use_fast_tokenizer, trust_remote_code=args.trust_remote_code)
model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device_map, torch_dtype=getattr(torch, args.torch_dtype))
if args.gradient_checkpointing:
    model.enable_input_require_grads()

# 加载、处理数据集和测试集
train_dataset_path = args.train_dataset_path
test_dataset_path = args.val_dataset_path

train_jsonl_new_path = args.train_formatted_path
test_jsonl_new_path = args.val_formatted_path

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 得到验证集
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

train_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps,
    logging_steps=args.logging_steps,
    num_train_epochs=args.num_train_epochs,
    save_steps=args.save_steps,
    learning_rate=args.learning_rate,
    save_on_each_node=args.save_on_each_node,
    gradient_checkpointing=args.gradient_checkpointing,
    report_to=args.report_to,
    run_name=args.run_name,
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

def main(args):
    trainer.train()
    test_df = pd.read_json(test_jsonl_new_path, lines=True)[:args.preview_samples_count]
    test_text_list = []
    for index, row in test_df.iterrows():
        instruction = row['instruction']
        input_value = row['input']
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]
        response = predict(messages, model, tokenizer)
        response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
        test_text_list.append(swanlab.Text(response_text))
        print(response_text)
    swanlab.log({"Prediction": test_text_list})
    swanlab.finish()

if __name__ == "__main__":
    main(args)