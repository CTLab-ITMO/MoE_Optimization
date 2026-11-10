# Goal of the experiment

The experiment applies the developed expert compression method (see the implementation in the common directory) to the "Qwen/Qwen1.5-MoE-A2.7B-Chat" model.

The experiment involves building new compressed layers from scratch based on the given model and datasets, integrating them into the architecture of the original model, and evaluating the changes in speed, in the memory of the compute device used while running the model, and in the accuracy of the model on the test set. Most results are saved as plots; the accuracy results of the models are saved as lists of values.

# Project structure

The experiment is implemented as a set of Python scripts using PyTorch and other libraries.

The files are named "N_*.py", where N is the sequence number of the script and corresponds to the order in which they are expected to be run.

The `N_*.py` scripts are edited directly and are the source of truth. Old `N_*.ipynb` notebooks (if any remain alongside) are legacy and are not regenerated from the scripts.

# Description of the experiment stages
Each step corresponds to one script (see the script with the corresponding number).

Three splits are used: train (fitting the compression and choosing the layer replacement order), validation (a held-out split for reconstruction, stages 1-8) and test (only the final accuracy measurement at stage 9). Previously the second split was test; now its role is played by validation.

## 1. Building the co-activation matrix

Creates data in data/activations/S_N: square matrices with size equal to the number of experts, storing the number of co-activations on the training and validation data.

## 2. Splitting the experts into pairs

Based on the co-activation matrix, an optimal split of the experts into pairs is built using a greedy algorithm.

## 3. Saving the experts

For each given MLP layer, its internal blocks, including every expert, are saved separately. This makes them more convenient to work with later.

## 4. Saving the input values

The training and validation data are passed through the model, and their values at the inputs of the MLP layers of interest are saved as a two-dimensional tensor.

## 5. Saving the data for each pair

For each pair, the input data on which it is activated are extracted. This allows the compression to be trained later only on the relevant data.

## 6. Compressing the experts

The direct application of the expert compression algorithm to the selected pairs based on the extracted input values.

## 7. Final statistics

Evaluates the layers individually on the training and validation data based on relative metrics.

## 8. Generation example

Shows several generation examples before and after compression. Allows assessing the differences of the final model in terms of the change in behaviour.

## 9. Test on real tasks

The model is tested on solving real tasks without reasoning, and the accuracy of its answers is evaluated. The order of sequential layer replacement is chosen on train (which has labels), and the final accuracy is measured on test.

# Data storage format

The experiment data are saved in the "data" and "plots" directories.

The "plots" directory stores the plots obtained during the experiment in the format "plots/TYPE/*.png", where TYPE is the type of the experiment.

The "data" directory stores the raw data for the model, in particular:
- data/activations/train_matrix_N - square matrices with size equal to the number of experts, storing the number of co-activations on the training data
- data/activations/validation_matrix_N - square matrices with size equal to the number of experts, storing the number of co-activations on the validation data
- data/pairs/N - the optimal split of the experts into pairs as a list of pairs, obtained from the training data
- data/experts/N/* - the components of MLP layer number N, including all experts: expert_K, gate, shared_expert, shared_expert_gate
- data/mlp_inputs/train_N - input data to the MLP block of layer N, a list of tensors
- data/mlp_inputs/validation_N - input data to the MLP block of layer N, a list of tensors
- data/activations/train_all_N - a list of tensors of size top_k, denoting the experts activated on the corresponding example
- data/activations/validation_all_N - a list of tensors of size top_k, denoting the experts activated on the corresponding example
- data/activations/pairs_train_N - input data to the MLP block of layer N for each pair of experts, a list of indices
- data/activations/pairs_validation_N - input data to the MLP block of layer N for each pair of experts, a list of indices
- data/fusion_experts/N/i - the compressed experts from pair i on layer N
