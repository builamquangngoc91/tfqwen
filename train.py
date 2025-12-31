import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ------------------------
# Model (Qwen2-VL)
# ------------------------
model_name = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Fix PAD token if missing
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# ✅ Use single GPU to avoid device_map auto sharding issues
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 0},   # ✅ force model on GPU 0
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
# Load small dataset (VQA-RAD)
# ------------------------
dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")
dataset = dataset.shuffle(seed=42)

# Take small dataset (≤ 2000)
dataset = dataset.select(range(min(2000, len(dataset))))

print("✅ Dataset loaded:", dataset)
print("✅ Example keys:", dataset[0].keys())

# ------------------------
# Convert VQA format -> Qwen2-VL Chat format
# ------------------------
def format_vqa(example):
    image = example["image"]
    question = example["question"]
    answer = example["answer"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question.strip()}
            ]
        },
        {"role": "assistant", "content": str(answer).strip()},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text, "image": image}

dataset = dataset.map(format_vqa, remove_columns=dataset.column_names)
print("✅ Dataset formatted:", dataset)

# ------------------------
# Data Collator (IMPORTANT)
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
        return_tensors="pt"
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    # ✅ MUST return CPU tensors (avoids pin_memory CUDA crash)
    return {k: v.cpu() for k, v in inputs.items()}

# ------------------------
# Training Arguments
# ------------------------
args = TrainingArguments(
    output_dir="qwen2vl-vqa-rad-lora-fp16",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none",

    # ✅ MUST
    remove_unused_columns=False,
    dataloader_pin_memory=False,
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

# ------------------------
# Train
# ------------------------
trainer.train()

# ------------------------
# Save
# ------------------------
model.save_pretrained("qwen2vl-vqa-rad-lora-fp16")
processor.save_pretrained("qwen2vl-vqa-rad-lora-fp16")

print("✅ Training complete. Saved to qwen2vl-vqa-rad-lora-fp16")