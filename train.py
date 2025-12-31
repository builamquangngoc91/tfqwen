import os
import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ------------------------
# Model (Qwen2-VL)
# ------------------------
model_name = "Qwen/Qwen2-VL-2B-Instruct"  # or "Qwen/Qwen2-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Fix pad token if missing
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# ------------------------
# LoRA Config
# ------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
print("✅ LoRA attached.")

# ------------------------
# Load Dataset
# ------------------------
dataset_name = "adamo1139/llava-instruct-150k-with-images"
dataset = load_dataset(dataset_name, split="train")

print("✅ Dataset loaded:", dataset)

# Dataset has:
# - image: PIL Image
# - conversations: list of dicts with keys {from, value}
# Image paths are inside zip; dataset card mentions two JSON formats.  [oai_citation:2‡Hugging Face](https://huggingface.co/datasets/adamo1139/llava-instruct-150k-with-images?utm_source=chatgpt.com)

# ------------------------
# Convert LLaVA Conversations -> Qwen2-VL Chat format
# ------------------------
def format_llava(example):
    image = example["image"]
    convs = example["conversations"]

    # LLaVA uses:
    # from: "human" / "gpt"
    # value: text, sometimes includes "<image>"
    qwen_messages = []

    for turn in convs:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"].replace("<image>", "").strip()

        if role == "user":
            # Qwen2-VL expects user content as list with image + text
            qwen_messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]
            })
        else:
            qwen_messages.append({
                "role": "assistant",
                "content": text
            })

    # Apply chat template (text only here; image passed separately in collator)
    chat_text = processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": chat_text, "image": image}

dataset = dataset.map(format_llava, remove_columns=dataset.column_names)
print("✅ Dataset converted.")

# ------------------------
# Collator: Multimodal batching
# ------------------------
MAX_LEN = 1024

def collate_fn(batch):
    texts = [x["text"] for x in batch]
    images = [x["image"] for x in batch]

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    # Move to model device
    return {k: v.to(model.device) for k, v in inputs.items()}

# ------------------------
# Training args
# ------------------------
args = TrainingArguments(
    output_dir="qwen2vl-llava-lora-fp16",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # 16 if GPU is small
    learning_rate=2e-4,
    num_train_epochs=1,              # dataset is big → start with 1 epoch
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none"
)

# ------------------------
# Trainer
# ------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    data_collator=collate_fn,
)

trainer.train()

# ------------------------
# Save
# ------------------------
model.save_pretrained("qwen2vl-llava-lora-fp16")
processor.save_pretrained("qwen2vl-llava-lora-fp16")
print("✅ Saved to qwen2vl-llava-lora-fp16")