"""
Download Jake's training data from Azure ML Studio to NVMe storage
"""
import os
import shutil

# Source path in Azure ML Studio - this is the exact path from our terminal!
SOURCE_PATH = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/jake-ai/code"

# Local NVMe path for faster training
NVME_PATH = "/mnt/nvme/jake_training_data"
LOCAL_FILE_PATH = os.path.join(NVME_PATH, "jake_training.jsonl")

def download_training_data():
    print("Current working directory:", os.getcwd())
    print("Listing source directory contents...")
    print(os.listdir(SOURCE_PATH))
    
    print("\nCreating NVMe directory...")
    os.makedirs(NVME_PATH, exist_ok=True)
    
    source_file = os.path.join(SOURCE_PATH, "jake_training.jsonl")
    print(f"\nCopying dataset from {source_file} to {LOCAL_FILE_PATH}...")
    
    shutil.copy2(source_file, LOCAL_FILE_PATH)
    print("Copy complete!")
    return LOCAL_FILE_PATH

if __name__ == "__main__":
    local_path = download_training_data()
    print(f"\nTraining data ready at: {local_path}")
