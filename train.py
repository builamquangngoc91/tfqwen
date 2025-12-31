import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# =========================================================
# 1) MODEL: Qwen2-VL
# =========================================================
model_name = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Ensure PAD token exists
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# Force model on 1 GPU to avoid weird device_map sharding issues
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# =========================================================
# 2) LORA
# =========================================================
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

# =========================================================
# 3) DATASET (small VQA dataset)
# =========================================================
dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")
dataset = dataset.shuffle(seed=42)

# take at most 2000 (train set has 1793)
dataset = dataset.select(range(min(2000, len(dataset))))

print("✅ Dataset loaded:", dataset)
print("✅ Example keys:", dataset[0].keys())

# =========================================================
# 4) FORMAT DATASET -> Qwen2-VL chat template
# =========================================================
MAX_TEXT_CHARS = 4000  # safety limit, avoids extremely long samples

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
            ],
        },
        {"role": "assistant", "content": str(answer).strip()},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Safety: prevent super long strings
    text = text[:MAX_TEXT_CHARS]

    return {"text": text, "image": image}

dataset = dataset.map(format_vqa, remove_columns=dataset.column_names)
print("✅ Dataset formatted:", dataset)

# =========================================================
# 5) COLLATOR (NO truncation!)
# =========================================================
def collate_fn(batch):
    texts = [x["text"] for x in batch]
    images = [x["image"] for x in batch]

    # IMPORTANT: DO NOT truncate here, otherwise Qwen2-VL image tokens mismatch
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=False,
        return_tensors="pt"
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    # IMPORTANT: Return CPU tensors only (avoids pin_memory CUDA crash)
    return {k: v.cpu() for k, v in inputs.items()}

# =========================================================
# 6) TRAINING ARGS
# =========================================================
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

    # ✅ MUST for multimodal + custom collator
    remove_unused_columns=False,

    # ✅ avoid pin_memory error if any tensor accidentally lands on GPU
    dataloader_pin_memory=False,
)

# =========================================================
# 7) TRAINER
# =========================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    data_collator=collate_fn,
)

# =========================================================
# 8) TRAIN
# =========================================================
trainer.train()

# =========================================================
# 9) SAVE
# =========================================================
save_dir = "qwen2vl-vqa-rad-lora-fp16"
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

print(f"✅ Training complete. Saved to {save_dir}")