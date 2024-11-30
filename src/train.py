"""
Jake AI Training Script for Azure ML
Optimized for A100 GPU training of Yi-34B
"""

import os
import logging
import argparse
from datetime import datetime

import torch
import yaml
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
from azureml.core import Run

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_dataset(json_file):
    """Load and format Jake's training data"""
    logger.info(f"Loading dataset from {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for conv in data['conversations']:
        for msg in conv['messages']:
            if msg['role'] == 'assistant':
                formatted_data.append({
                    'text': f"<|im_start|>system\nYou are Jake, an unhinged door-to-door salesman known for your epic MLM stories and wild sales adventures.<|im_end|>\n<|im_start|>user\nTell me a Jake story<|im_end|>\n<|im_start|>assistant\n{msg['content']}<|im_end|>"
                })
    
    logger.info(f"Prepared {len(formatted_data)} training examples")
    return Dataset.from_list(formatted_data)

def setup_model_and_tokenizer(model_name, quantization_config):
    """Initialize Yi-34B with memory optimizations"""
    logger.info(f"Loading model: {model_name}")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_lora_config(config):
    """Configure LoRA for efficient fine-tuning"""
    return LoraConfig(
        r=config['model']['lora_r'],
        lora_alpha=config['model']['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def setup_training_arguments(config, run):
    """Configure training parameters"""
    output_dir = os.path.join(run.get_output_dir(), 'checkpoints')
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        fp16=True,
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        report_to="azure"
    )

class AzureMLCallback:
    """Custom callback for Azure ML logging"""
    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                self.run.log(key, value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()

    # Get Azure ML run context
    run = Run.get_context()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        config['model']['name'],
        config['model']['quantization']
    )
    
    # Load dataset
    dataset = prepare_dataset(os.path.join(args.data_dir, 'jake_training.json'))
    
    # Setup LoRA
    lora_config = setup_lora_config(config)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Setup training arguments
    training_args = setup_training_arguments(config, run)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[AzureMLCallback(run)]
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    output_dir = os.path.join(run.get_output_dir(), 'final_model')
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
