import os
import yaml
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

def download_model(repo_id="Avencast/EveNet", filename="checkpoints.20M.a4.last.ckpt"):
    """Download pretrained model from Hugging Face Hub"""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=os.getenv("HF_TOKEN", None),
        cache_dir=os.getenv("EVENET_MODEL_PATH", None)
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
            force=True
        )
    
    # Save updated config to temp file
    temp_config = f"./{os.path.basename(user_config_path)}_updated.yaml"
    with open(temp_config, "w") as f:
        OmegaConf.save(conf, f)
    
    return temp_config
