import argparse
import os
import subprocess
import sys
from .utils import download_model, prepare_config, download_demo_data

def main():
    parser = argparse.ArgumentParser(
        description="Download EveNet model and launch fine-tuning"
    )
    parser.add_argument(
        "config", 
        nargs="?",
        default="evenet_tools/share/finetune-example.yaml",
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--ray_dir",
        default="~/ray_results",
        help="Ray results directory (default: ~/ray_results)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Download demo dataset and run with default config"
    )
    args = parser.parse_args()

    # Handle demo mode
    if args.demo:
        print("🚀 Demo mode: downloading dataset and launching with default config")

        # Download demo dataset
        print("⏳ Downloading demo dataset files...")
        download_demo_data()
        print("✅ Demo dataset ready")

        # Use default config for demo
        args.config = "evenet_tools/share/finetune-example.yaml"

    # Step 1: Download model
    print("⏳ Checking for pretrained model...")
    ckpt_path = download_model()
    print(f"✅ Model ready at: {ckpt_path}")

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
