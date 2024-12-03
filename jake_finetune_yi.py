"""
Jake AI Fine-tuning using Yi-34B
This script contains the core training logic that we'll use in Colab
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json

def prepare_dataset(json_file):
    """Load and format Jake's training data"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for conv in data['conversations']:
        for msg in conv['messages']:
            if msg['role'] == 'assistant':
                # Format optimized for Yi-34B's instruction format
                formatted_data.append({
                    'text': f"<|im_start|>system\nYou are Jake, an unhinged door-to-door salesman known for your epic MLM stories and wild sales adventures.<|im_end|>\n<|im_start|>user\nTell me a Jake story<|im_end|>\n<|im_start|>assistant\n{msg['content']}<|im_end|>"
                })
    
    return Dataset.from_list(formatted_data)

def setup_model_and_tokenizer():
    """Initialize Yi-34B with memory optimizations"""
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        "01-ai/Yi-34B",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_lora_config():
    """Configure LoRA for efficient fine-tuning"""
    return LoraConfig(
        r=64,  # Increased rank for better capacity
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def setup_training_arguments(output_dir):
    """Configure training parameters optimized for Colab"""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Small batch size for memory efficiency
        gradient_accumulation_steps=16,  # Compensate for small batch size
        learning_rate=1e-4,
        fp16=True,
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        report_to="none"  # Disable wandb logging
    )

def create_jake_trainer(model, tokenizer, dataset, training_args):
    """Initialize the trainer with all components"""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

def generate_jake_story(model, tokenizer, prompt="Tell me a Jake story"):
    """Test the model with a sample prompt"""
    formatted_prompt = f"<|im_start|>system\nYou are Jake, an unhinged door-to-door salesman known for your epic MLM stories and wild sales adventures.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=1000,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
