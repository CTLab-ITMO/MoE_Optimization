import os
import subprocess
import shutil
import sys

sys.path.append("../..")
from MoE_Optimization.Qwen_MoE_experts_fusion_experiment.config import DATA_DIR, MODEL_NAME

os.environ["HF_HUB_CACHE"] = "/glazkov-dev/.cache"

scripts = ["1_generate_activation_matrix.py", "3_save_experts.py", "4_write_mlp_inputs.py", "5_activations_all.py"]

for s in scripts:
    print("Running", s)
    subprocess.run(["python", s], check=True)

#make backup of model weights
os.makedirs(f"/glazkov-dev/{MODEL_NAME}", exist_ok=True)
shutil.copytree(f"{DATA_DIR}", f"/glazkov-dev/{MODEL_NAME}", dirs_exist_ok=True)
