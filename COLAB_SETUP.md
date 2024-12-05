# Setting up Jake AI Training on Google Colab

## Prerequisites
1. Google account with access to Google Colab
2. Your training data (`jake_training.json`)
3. Google Drive with at least 20GB free space

## Setup Steps

1. **Create Google Drive Structure**
   ```
   jake-ai/
   ├── data/
   │   └── training/
   │       └── jake_training.json
   ├── checkpoints/
   └── output/
   ```

2. **Colab Setup**
   - Open Google Colab
   - Create a new notebook
   - Connect to a GPU runtime (Runtime > Change runtime type > GPU)
   - Mount Google Drive

3. **Install Required Packages**
   ```python
   !pip install -q torch==2.1.0 transformers==4.35.0 accelerate==0.24.0
   !pip install -q bitsandbytes==0.41.1 scipy
   !pip install -q git+https://github.com/huggingface/peft.git
   ```

4. **Mount Drive & Setup**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Copy training script
   !mkdir -p /content/jake-ai
   !cp "/content/drive/MyDrive/jake-ai/jake_finetune_yi.py" /content/jake-ai/
   ```

5. **Import and Run Training**
   ```python
   import sys
   sys.path.append('/content/jake-ai')
   from jake_finetune_yi import *
   
   # Setup model
   model, tokenizer = setup_model_and_tokenizer()
   
   # Load dataset
   dataset = prepare_dataset("/content/drive/MyDrive/jake-ai/data/training/jake_training.json")
   
   # Setup LoRA
   lora_config = setup_lora_config()
   model = prepare_model_for_kbit_training(model)
   model = get_peft_model(model, lora_config)
   
   # Setup training
   training_args = setup_training_arguments("/content/jake-ai-checkpoints")
   trainer = create_jake_trainer(model, tokenizer, dataset, training_args)
   
   # Start training
   trainer.train()
   
   # Save final model
   trainer.save_model("/content/drive/MyDrive/jake-ai/output/jake-yi-final")
   ```

6. **Test Generation**
   ```python
   # Load the trained model
   story = generate_jake_story(model, tokenizer)
   print(story)
   ```

## Important Notes

1. **Memory Management**
   - The script is optimized for Colab's T4 GPU
   - Uses 4-bit quantization and gradient checkpointing
   - Small batch size with gradient accumulation

2. **Checkpointing**
   - Models are saved every 50 steps
   - Only keeps last 3 checkpoints to save space
   - Final model saved to Drive

3. **Disconnection Handling**
   - Regular checkpoints allow resuming training
   - Keep Colab tab active to prevent disconnections
   - Use "Connect to hosted runtime" if disconnected

4. **Monitoring**
   - Watch GPU memory usage with `nvidia-smi`
   - Check training progress in output logs
   - Monitor Drive space during training

## Troubleshooting

1. **Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation steps
   - Clear notebook runtime and restart

2. **Drive Space**
   - Clean up old checkpoints
   - Monitor space usage
   - Keep at least 10GB free

3. **Slow Training**
   - Check GPU allocation
   - Ensure no other notebooks using GPU
   - Close unnecessary browser tabs
