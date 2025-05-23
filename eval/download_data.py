from datasets import load_dataset
from pathlib import Path

def download_diabetes_dataset(download_dir="./eval"):
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("abdelhakimDZ/diabetes_QA_dataset", cache_dir=download_dir, trust_remote_code=True)
    return dataset


if __name__ == "__main__":
    download_diabetes_dataset("./hitrate")