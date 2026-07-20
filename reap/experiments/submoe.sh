#!/bin/bash
# TODO: Parse observer files
model_name=${1}
dataset_name=${2}
compression_ratio=${3}
server_log_file_name=${4}
port=${5}
seed=${6:-42}
# qa
run_lm_eval=${7:-true}
# coding
run_evalplus=${8:-true}
run_livecodebench=${9:-true}
# math
run_math=${10:-false}
# wildbench
run_wildbench=${11:-false}
singleton_super_experts=${12:-false}
singleton_outlier_experts=${13:-false}

echo "Running SubMoE: $model_name, dataset: $dataset_name, compression ratio: $compression_ratio, seed ${seed}, super expert singletons: ${singleton_super_experts}, outlier expert singletons: ${singleton_outlier_experts}"

_merge_tmpfile=$(mktemp)
python src/reap/main.py \
    --compression-ratio ${compression_ratio} \
    --model-name ${model_name} \
    --dataset-name ${dataset_name} \
    --profile false \
    --server_log_file_name $server_log_file_name \
    --vllm-port $port \
    --expert-sim online_characteristic_activation_dist \
    --distance_measure cosine \
    --linkage-method average \
    --cluster-method kmeans \
    --frequency-penalty false \
    --multi-layer 2 \
    --merge-method submoe \
    --output-file-name observations_1024_cosine-seed_${seed}.pt \
    --merged-model-dir-name "submoe-${compression_ratio}" \
    --cluster-description "submoe" \
    --samples-per-category 1024 \
    --do-eval false 2>&1 | tee "$_merge_tmpfile"

# Extract the actual merged model directory from Python output
merged_model=$(grep '^MERGED_MODEL_DIR=' "$_merge_tmpfile" | tail -1 | cut -d'=' -f2-)
rm -f "$_merge_tmpfile"

if [[ -z "$merged_model" ]]; then
    # Fallback: reconstruct the path if Python didn't print it
    short_model_name=$(echo $model_name | cut -d'/' -f2)
    short_dataset_name=$(echo $dataset_name | cut -d'/' -f2)
    merged_model="artifacts/${short_model_name}/${short_dataset_name}/non_uniform_merged_models/submoe-${compression_ratio}/submoe"
    echo "WARNING: Could not extract merged model dir from Python output, using fallback: ${merged_model}"
fi

bash experiments/eval.sh ${merged_model} ${seed} ${port} ${server_log_file_name} ${run_lm_eval} ${run_evalplus} ${run_livecodebench} ${run_math} ${run_wildbench}
echo "Finished SubMoE for model: ${model_name}, dataset: ${dataset_name}, compression ratio: ${compression_ratio}"
echo "Removing safetensor files from ${merged_model}"
# rm ${merged_model}/*.safetensors
