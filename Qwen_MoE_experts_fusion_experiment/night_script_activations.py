import os
import subprocess
import shutil
import sys
sys.path.append("../..")
from MoE_Optimization.Qwen_MoE_experts_fusion_experiment.config import DATA_DIR, MODEL_NAME


CLUSTERING_ALGORITHMS = [
    'create_clusters_by_PI',
    'create_clusters_by_EC',    
    'create_clusters_by_PI_greedy_aglomerative',
    'create_clusters_by_EC_greedy_aglomerative']
# CLUSTERING_ALGORITHMS = ["create_clusters_by_EC_greedy_aglomerative"]

ACTIVATION_RELATED_DIRECTORIES = [
    'activations/pairs',
    'fusion_experts',
    'pairs'
]

os.environ["HF_HUB_CACHE"] = "/glazkov-dev/.cache"

scripts = [ '2_find_best_matches.py', "5_activations.py", "6_fusion.py", "9_test.py"]
# scripts = [ "5_activations.py", "6_fusion.py", "9_test.py"]

for algo in CLUSTERING_ALGORITHMS:
    env = os.environ.copy()
    env["PAIRING_METHOD"] = algo
    for s in scripts:
        print("Running", s)
        subprocess.run(["python", s], env=env, check=True)

    #make backup
    for d in ACTIVATION_RELATED_DIRECTORIES:
        os.makedirs(f"/glazkov-dev/{MODEL_NAME}/{algo}", exist_ok=True)
        shutil.copytree(f"{DATA_DIR}/{d}", f"/glazkov-dev/{MODEL_NAME}/{algo}/{d}", dirs_exist_ok=True)
    