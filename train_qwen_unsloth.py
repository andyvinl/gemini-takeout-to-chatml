# Fine-tune Qwen with Unsloth (LoRA) on qwen_chatml_ready.jsonl, then export GGUF for Ollama.
# Run on the EC2 instance inside the `unsloth` conda env:
#   python qwentrain.py          # step 1: builds qwen_chatml_ready.jsonl
#   python train_qwen_unsloth.py # step 2: trains + exports
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"   # swap for a bigger/smaller Qwen if you like
DATA_FILE  = "qwen_chatml_ready.jsonl"
MAX_LEN    = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.map(lambda ex: {
    "text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)
})
print(f"Training on {len(dataset)} conversations")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_LEN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,          # small dataset -> a few passes
        learning_rate=2e-4,
        warmup_steps=5,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
trainer.train()

# Save the LoRA adapter first so a GGUF failure can't lose the training.
model.save_pretrained("qwen_andy_lora")
tokenizer.save_pretrained("qwen_andy_lora")
print("LoRA adapter saved to ~/qwen_andy_lora")

# Merge + export to GGUF (Q4_K_M is a good size/quality balance for a laptop).
model.save_pretrained_gguf("qwen_andy_gguf", tokenizer, quantization_method="q4_k_m")
print("GGUF saved to ~/qwen_andy_gguf")
