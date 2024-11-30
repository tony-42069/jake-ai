# Setting up Jake AI Training on Azure ML

## Prerequisites
1. Azure Account (with free $200 credit for new accounts)
2. Training data (`jake_training.json`)
3. Azure ML Workspace

## Azure Setup Steps

### 1. Create Azure ML Workspace
1. Go to Azure Portal (portal.azure.com)
2. Create new "Machine Learning" resource
3. Select your subscription and resource group
4. Choose a workspace name (e.g., "jake-ai-workspace")
5. Select your region (choose one with GPU availability)

### 2. Configure Compute Instance
For Yi-34B, we recommend:
- VM Size: NC24ads_A100_v4 (1x A100 80GB GPU)
  - This will easily handle Yi-34B with room for training
  - Costs ~$3.60/hour but training will be much faster
- Alternative: NC6s_v3 (1x V100 16GB GPU)
  - More cost-effective (~$0.90/hour)
  - Will require more optimization but still works

### 3. Project Structure
```
jake-ai/
├── src/
│   ├── train.py           # Training script
│   └── utils.py           # Helper functions
├── data/
│   └── training/
│       └── jake_training.json
├── config/
│   └── training_config.yaml
└── README.md
```

### 4. Environment Setup
Create a new compute instance with these specs:
- Python 3.10
- PyTorch 2.1.0+cu118
- Required packages in requirements.txt

### 5. Data Upload
1. Go to "Data" in Azure ML Studio
2. Create new "Data Asset"
3. Upload your training data
4. Register as "jake-training-data"

### 6. Training Configuration
Use these settings in your YAML config:
```yaml
compute:
  instance_type: "NC24ads_A100_v4"
  
training:
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  num_epochs: 3
  
model:
  name: "01-ai/Yi-34B"
  quantization: "4bit"
  lora_r: 64
  lora_alpha: 128
```

### 7. Monitoring
- Use Azure ML Studio's monitoring dashboard
- Track:
  - Training loss
  - GPU utilization
  - Memory usage
  - Training progress

## Cost Management
1. **Estimated Costs**
   - A100: ~$3.60/hour
   - V100: ~$0.90/hour
   - Storage: Minimal (~$0.08/GB/month)

2. **Cost Optimization**
   - Use spot instances when possible (up to 90% cheaper)
   - Delete compute instance when not in use
   - Clean up unused storage
   - Monitor usage with Azure Cost Management

## Important Notes

### Security
1. Store credentials securely using Azure Key Vault
2. Use managed identities for authentication
3. Set up proper access controls

### Best Practices
1. Enable early stopping to prevent wasted compute
2. Use checkpointing every 30 minutes
3. Monitor training metrics
4. Keep logs for troubleshooting

### Troubleshooting
1. **Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation
   - Check GPU memory usage

2. **Training Issues**
   - Check logs in Azure ML Studio
   - Monitor GPU utilization
   - Verify data loading

3. **Cost Overruns**
   - Set up budget alerts
   - Monitor usage regularly
   - Use auto-shutdown policies

## Next Steps
1. Set up Azure ML Workspace
2. Configure compute instance
3. Upload training data
4. Run training script
5. Monitor progress and costs

Need help with any specific step? Let me know!
