# Jake AI Training Steps

## 1. Start Compute (2 minutes)
- Go to Azure ML Studio
- Compute > Compute Instances
- Start `jake-ai-training` instance
- Wait for status "Running"

## 2. Upload Data (3 minutes)
- Click "Data" in sidebar
- Create new datastore
- Upload `jake_training.jsonl`
- Name it "jake-training-data"

## 3. Start Training (5 minutes)
- Open JupyterLab
- Upload training files
- Run training script
- Verify training started

## 4. Monitor Training (2-3 hours)
- Watch training metrics
- Check loss values
- Monitor GPU usage

## 5. Save Model
- Training auto-saves checkpoints
- Final model saved automatically
- Copy to your storage

## Important Notes
- Estimated cost: ~$10.80
- Training time: 2-3 hours
- Auto-shutdown enabled
- Checkpoints every 50 steps

## Troubleshooting
If training stops:
1. Check GPU memory usage
2. Verify data loading
3. Check training logs
4. Restart from checkpoint if needed

## Cost Saving Tips
- Start instance right before training
- Use auto-shutdown
- Monitor training progress
- Stop instance if not needed

## Success Metrics
- Loss should decrease steadily
- Check generated text quality
- Save best checkpoint

Remember: You can monitor training remotely - no need to keep your computer on!
