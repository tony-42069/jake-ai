import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

def prepare_dataset(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for conv in data['conversations']:
        for msg in conv['messages']:
            if msg['role'] == 'assistant':
                formatted_data.append({
                    'text': f"<s>[INST] Tell me a Jake story [/INST] {msg['content']}</s>"
                })
    
    return Dataset.from_list(formatted_data)

def create_peft_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

def main():
    # Load base model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"  # or "mistralai/Mistral-7B-v0.1"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare dataset
    dataset = prepare_dataset("../data/training/jake_training.json")
    
    # Create PEFT config and wrap model
    peft_config = create_peft_config()
    model = get_peft_model(model, peft_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="jake-ai-model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_steps=50,
        logging_steps=10,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train
    trainer.train()
    
    # Save the final model
    trainer.save_model("jake-ai-final")

if __name__ == "__main__":
    main()
