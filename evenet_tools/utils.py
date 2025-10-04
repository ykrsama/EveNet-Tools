import os
import yaml
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from omegaconf import OmegaConf

def download_model(repo_id="Avencast/EveNet", filename="checkpoints.20M.a4.last.ckpt"):
    """Download pretrained model from Hugging Face Hub"""
    cache_dir = os.getenv("EVENET_MODEL_PATH", None)
    token = os.getenv("HF_TOKEN", None)

    # Check if model already exists locally
    cached_file = try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir
    )

    if cached_file is not None:
        print(f"🔄 Model already exists at: {cached_file}")
        return cached_file

    # Download if not cached
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        cache_dir=cache_dir
    )

def prepare_config(user_config_path: str, model_checkpoint: str):
    """Update config with downloaded model path"""
    # Load user config with merging base configs
    conf = OmegaConf.load(user_config_path)
    
    # Apply model checkpoint path
    if not OmegaConf.select(conf, "options.Training.pretrain_model_load_path"):
        OmegaConf.update(
            conf,
            "options.Training.pretrain_model_load_path",
            model_checkpoint,
            #force=True
        )
    
    # Save updated config to temp file
    temp_config = f"./{os.path.basename(user_config_path)}_updated.yaml"
    with open(temp_config, "w") as f:
        OmegaConf.save(conf, f)
    
    return temp_config

def download_demo_data(cache_dir="cache/data"):
    """Download demo dataset files from Hugging Face Hub"""
    os.makedirs(cache_dir, exist_ok=True)

    repo_id = "Avencast/EveNet"
    files_to_download = [
        "data_Combined_Balanced_run_0.parquet",
        "normalization.pt"
    ]

    downloaded_files = []

    for filename in files_to_download:
        file_path = os.path.join(cache_dir, filename)

        # Check if file already exists
        if os.path.exists(file_path):
            print(f"🔄 File already exists: {file_path}")
            downloaded_files.append(file_path)
            continue

        # Download file
        print(f"⏳ Downloading {filename}...")
        try:
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=cache_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ Downloaded: {downloaded_file}")
            downloaded_files.append(downloaded_file)
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            raise

    return downloaded_files
