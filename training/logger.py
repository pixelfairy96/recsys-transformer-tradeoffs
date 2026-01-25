import os
import json
import yaml
import torch
import subprocess
from datetime import datetime


def create_run_dir(base_dir="logs"):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir


def save_config(config, run_dir):
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)


def get_env_info():
    info = {}

    info["pytorch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda

        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                encoding="utf-8"
            )
            info["nvidia_driver"] = result.strip()
        except Exception:
            info["nvidia_driver"] = "unknown"

    return info


def save_env_info(env_info, run_dir):
    with open(os.path.join(run_dir, "env.txt"), "w") as f:
        for k, v in env_info.items():
            f.write(f"{k}: {v}\n")


def save_metrics(metrics, run_dir):
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)