import argparse
import subprocess
import sys
from .utils import download_model, prepare_config

def main():
    parser = argparse.ArgumentParser(
        description="Download EveNet model and launch fine-tuning"
    )
    parser.add_argument(
        "config", 
        nargs="?",
        default="share/finetune-example.yaml",
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--ray_dir", 
        default="~/ray_results",
        help="Ray results directory (default: ~/ray_results)"
    )
    args = parser.parse_args()

    # Step 1: Download model
    print("⏳ Downloading pretrained model...")
    ckpt_path = download_model()
    print(f"✅ Model downloaded to: {ckpt_path}")

    # Step 2: Update config
    print("🔄 Preparing training configuration...")
    updated_config = prepare_config(args.config, ckpt_path)
    print(f"⚙️ Using updated config: {updated_config}")

    # Step 3: Launch training
    print("🚀 Launching fine-tuning...")
    cmd = [
        "evenet-train" if os.name != "nt" else "evenet-train.exe",
        updated_config,
        "--ray_dir",
        args.ray_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
