from functools import partial
from typing import Iterable, Literal
from MoE_Optimization.common.expert_logger import ExpertsLogger
from MoE_Optimization.common.expert_logger_data_converter import ExpertLoggerDataConverter
from MoE_Optimization.common.expert_matcher import ExpertMatcher
from MoE_Optimization.common.moe_utils import MoeHandle, discover_moe_layers
import torch


class ExpertClusterer:
    def __init__(self, model: torch.nn.Module, example_inputs: Iterable, ):
        self.module2name = {m: n for n, m in model.named_modules()}
        self.model = model
        self.example_inputs = example_inputs
        self.experts_logger = partial(ExpertsLogger, model=model)
        self.moe_handles: list[MoeHandle] = discover_moe_layers(model)


    def create_clusters_by_PI(self, target_layers: list[MoeHandle]):
        '''Gather stats from model on it's gate, transform it to matrix of PI metric and make divisive clustering of that matrix to minimize PI
        
        Returns:
            moehandles2clusters: dict of layer_idx -> list of pairs or triplets of experts with minimal PI to merge
        '''
        layer_idx2clusters = {}
        name2activations = self._get_activations_from_gate(target_layers)
        for target_layer_handler in target_layers:
            layer_gate_name = self.module2name[target_layer_handler.gate]
            gate_activations = name2activations[layer_gate_name]
            clusters, _ = self.create_clusters_by_PI_one_layer(target_layer_handler, gate_activations, log_history=False)
            layer_idx2clusters[target_layer_handler.layer_idx] = clusters
        return layer_idx2clusters
    
    def create_clusters_by_PI_one_layer(self, target_layer: MoeHandle, gate_activations: torch.Tensor, log_history=False):
        gate_PI_matrix = ExpertLoggerDataConverter.compute_PI_matrix(self.model.num_experts, 
                                                                     gate_activations, 
                                                                     k = target_layer.gate.top_k)
        clusters, history = ExpertMatcher.find_by_dividive_clustering(gate_PI_matrix, 
                                                                maximize_metric=False, 
                                                                log_history=log_history)
        return clusters, history

    def _get_activations_from_gate(self, target_layers: list[MoeHandle]) -> dict[str, torch.Tensor]:
        name2activations = self.get_activations(target_layers, attach_to_gate_layer=True, attach_to_expert_layer=False)
        ExpertLoggerDataConverter._transform_activations_for_analysis(name2activations)
        return name2activations
    

    @staticmethod
    def _get_aggregated_embedding(E: torch.Tensor, mode: Literal['pca', 'mean']):
        """
        E is a embedding matrix / vector for one expert on one layer (!). 
        E size is: [n_observations x embedding_size]
        if E is a vector it would be treated as [1 x embbedding_size] matrix. 
        """
        if E.ndim == 1:
            E = E.reshape(1, -1)
        elif E.ndim > 2:
            raise ValueError('Expected vector or 2D matrix')

        if mode == 'mean':
            return torch.mean(E, dim=0)
        elif mode == 'svd':
            m = E.mean(dim=0)
            E -= m
            U, S, Vt = torch.linalg.svd(E.T, full_matrices=False)
            one_rank_emb = U[:, 0] * S[0]
            one_rank_emb += m
            return one_rank_emb
        else:
            raise ValueError('Expected for `mode` one of: `pca`, `mean`')

    def get_activations(self,
                        target_layers: list[MoeHandle], 
                        attach_to_gate_layer=True, 
                        attach_to_expert_layer=False
                        ) -> dict[str, list[torch.Tensor]]:
        '''
        Returns:
            dict: layer name (model.layers.0.mlp.experts.0) and list of it's activation tensors (1D)
            for activation - list of H-shaped vectors for exact expert for every forward. For gate - list of N_EXPERTS_shaped for every forward.
        '''

        if not (isinstance(target_layers, (list, tuple)) and len(target_layers)):
            raise TypeError(
                'Expected non-empty list or tuple of MoEHandle or int')

        layer_names = []
        if (attach_to_gate_layer):
            gate_names = [
                self.module2name[moe_handle.gate] for moe_handle in target_layers
            ]
            layer_names += gate_names
        if (attach_to_expert_layer):
            experts_names = [self.module2name[expert]
                             for handle in target_layers for expert in handle.children()]
            layer_names += experts_names
        return self._get_activations(layer_names)

    @torch.no_grad()
    def _get_activations(self, layer_names: list[str]) -> dict[str, list[torch.Tensor]]:
        experts_logger: ExpertsLogger = self.experts_logger(
            layer_of_interest=layer_names)
        experts_logger.reset()
        name2activation = experts_logger.get_activations(self.example_inputs)
        return name2activation
