# Jake AI Troubleshooting Guide

## Storage and Space Issues

### Disk Space Management
When training Jake on Azure ML, you might encounter disk space issues. Here's how to handle them:

1. **Check Available Space**
```bash
df -h
```

2. **Configure HuggingFace Cache**
Set environment variables to use Azure file share storage:
```bash
export HF_HOME=/mnt/batch/tasks/shared/LS_root/mounts/clusters/jake-ai/code/huggingface
export TRANSFORMERS_CACHE=/mnt/batch/tasks/shared/LS_root/mounts/clusters/jake-ai/code/huggingface/transformers-cache
```

3. **Create Cache Directory**
```bash
mkdir -p /mnt/batch/tasks/shared/LS_root/mounts/clusters/jake-ai/code/huggingface/transformers-cache
```

### Common Issues and Solutions

1. **"No space left on device" Error**
   - This usually means the model is downloading to the wrong location
   - Solution: Use the Azure file share (100TB available) instead of local storage
   - Make sure environment variables are set before starting training

2. **Model Downloads After Restart**
   - Normal behavior when compute instance restarts
   - Downloads go to configured HuggingFace cache location
   - Ensure enough space in target directory

3. **Storage Location Priority**
   - Azure file share: Preferred for model storage (100TB available)
   - Local storage: Limited and cleared on restart
   - NVMe: May not be available/properly mounted

## Best Practices

1. **Before Starting Training**
   - Check available space with `df -h`
   - Set environment variables for HuggingFace cache
   - Verify cache directory exists
   - Ensure you're in correct project directory

2. **During Training**
   - Monitor disk usage
   - Watch for warning messages about space
   - Check training logs for errors

3. **After Instance Restart**
   - Reset environment variables
   - Expect model to download again
   - Verify storage configuration

## Cost Management
- Instance restarts clear local storage
- Model re-downloads are necessary after restarts
- Use Azure file share to avoid repeated downloads in the future

## Quick Reference
```bash
# Check disk space
df -h

# Set up storage
export HF_HOME=/mnt/batch/tasks/shared/LS_root/mounts/clusters/jake-ai/code/huggingface
export TRANSFORMERS_CACHE=/mnt/batch/tasks/shared/LS_root/mounts/clusters/jake-ai/code/huggingface/transformers-cache
mkdir -p $TRANSFORMERS_CACHE

# Verify project directory
cd ~/cloudfiles/code/Users/dsadellari/jake-ai
```
