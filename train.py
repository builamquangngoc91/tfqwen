import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ------------------------
# Model
# ------------------------
model_name = "Qwen/Qwen2-7B-Instruct"   # you can switch to Qwen2-1.5B for smaller GPU

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
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

print("Initializing LoRA parameters...")


dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# ------------------------
# Convert Dolly -> Qwen chat format
# ------------------------
def format_dolly(example):
    instruction = example["instruction"]
    context = example.get("context", "")
    response = example["response"]

    user_msg = instruction.strip()
    if context and len(context.strip()) > 0:
        user_msg += "\n\nContext:\n" + context.strip()

    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": response.strip()},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

# ------------------------
# Tokenize dataset (old TRL-friendly)
# ------------------------
MAX_LEN = 1024  # ✅ small dataset -> 1024 is enough; use 2048 if GPU is strong

def tokenize_fn(example):
    text = format_dolly(example)
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

print("dataset tokenized.")


# ------------------------
# Training args (FP16)
# ------------------------
args = TrainingArguments(
    output_dir="qwen-dolly-lora-fp16",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none"
)


trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=args,
)

trainer.train()

model.save_pretrained("qwen-dolly-lora-fp16")
tokenizer.save_pretrained("qwen-dolly-lora-fp16")
