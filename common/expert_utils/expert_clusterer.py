from functools import partial
from typing import Callable, Iterable, Literal

from MoE_Optimization.common.expert_utils.expert_logger import ExpertsLogger
from MoE_Optimization.common.expert_utils.expert_logger_data_converter import ExpertLoggerDataConverter
from MoE_Optimization.common.expert_utils.expert_matcher import ExpertMatcher
from MoE_Optimization.common.moe_utils import MoeHandle, discover_moe_layers
import MoE_Optimization.common.clustering_utils as clustering_utils
import re
import torch
import pandas as pd
import numpy as np


class ExpertClusterer:
    def __init__(self, model: torch.nn.Module, example_inputs: Iterable, ):
        self.module2name = {m: n for n, m in model.named_modules()}
        self.model = model
        self.example_inputs = example_inputs
        self.experts_logger = partial(ExpertsLogger, model=model)
        self.moe_handles: list[MoeHandle] = discover_moe_layers(model)
    
    def _create_clusters_by(self, create_clusters_one_layer_func: Callable, target_layers: list[MoeHandle], **kwargs):
        '''Universal method for creating clusters for each of supplied layers'''
        layer_idx2clusters = {}
        name2activations = self._get_activations_from_gate(target_layers)
        for target_layer_handler in target_layers:
            layer_gate_name = self.module2name[target_layer_handler.gate]
            gate_activations = name2activations[layer_gate_name]
            
            clusters, _ = create_clusters_one_layer_func(target_layer_handler, gate_activations, log_history=False, **kwargs)
            
            layer_idx2clusters[target_layer_handler.layer_idx] = clusters
        return layer_idx2clusters
    
    
    def create_clusters_by_front_PI_and_max_Calinski(self, target_layers: list[MoeHandle]):
        return self._create_clusters_with_embeddings(
            self.create_clusters_by_PI_front_points,
            target_layers
        )
    
    #TODO make better
    def _create_clusters_with_embeddings(self, find_front_points_func: Callable, target_layers: list[MoeHandle], **kwargs):
        '''Universal method for creating clusters for each of supplied layers, with additional info about expert's activations (embeddings)'''
        layer_idx2clusters = {}
        name2activations = self._get_all_activations(target_layers)
        for target_layer_handler in target_layers:
            layer_gate_name = self.module2name[target_layer_handler.gate]
            gate_activations = name2activations[layer_gate_name]
            
            
            interesting_steps, history = find_front_points_func(target_layer_handler, gate_activations, log_history=False, **kwargs)
            
            layer_expert_df = self._get_tasks_expert_embeddings(name2activations, layers=[target_layer_handler.layer_idx])
            
            step_PI_and_calinksi_score_df = clustering_utils.build_calinski_score_df(
                history, 
                layer_expert_df.drop(['layer', 'expert'], axis=1).to_numpy(dtype=np.float64), 
                row_experts = layer_expert_df['expert'].to_numpy(),
                some_popularity_metric_column="total_PI_after_split")
            
            best_step = step_PI_and_calinksi_score_df['calinski_harabasz'][interesting_steps].idxmax()
            clusters = history[int(best_step)]['all_clusters']
            
            layer_idx2clusters[target_layer_handler.layer_idx] = clusters
        return layer_idx2clusters
    
    def create_clusters_by_PI_one_layer(self, target_layer: MoeHandle, gate_activations: torch.Tensor, log_history=False, maximize_metric=False, **kwargs):
        '''Gets last clustering of dividive clustering'''
        clusters, history = self._create_clusters_by_metric_by_clustering_algo(
            ExpertMatcher.find_by_dividive_clustering, 
            ExpertLoggerDataConverter.compute_PI_matrix,
            target_layer, 
            gate_activations, 
            log_history=log_history, 
            maximize_metric=maximize_metric,
            **kwargs
            )
        return clusters, history
    
    def create_clusters_by_EC_one_layer(self, target_layer: MoeHandle, gate_activations: torch.Tensor, log_history=False, maximize_metric=True, **kwargs):
        '''Gets last clustering of dividive clustering'''
        clusters, history = self._create_clusters_by_metric_by_clustering_algo(
            ExpertMatcher.find_by_dividive_clustering, 
            ExpertLoggerDataConverter.compute_EC_matrix, 
            target_layer, 
            gate_activations, 
            log_history=log_history, 
            maximize_metric=maximize_metric,
            **kwargs
            )
        return clusters, history
    
    def create_clusters_by_PI_but_find_max_on_EC(self, target_layer: MoeHandle, gate_activations: torch.Tensor, **kwargs):
        '''Gets best clustering of dividive clustering watching on EC'''
        clusters, history = self._create_clusters_by_two_metric_by_clustering_algo(
            ExpertMatcher.find_by_dividive_clustering_PI_and_max_EC,
            ExpertLoggerDataConverter.compute_PI_matrix,
            ExpertLoggerDataConverter.compute_EC_matrix, 
            target_layer, 
            gate_activations, 
            maximize_metric=False
            )
        return clusters, history
    
    def create_clusters_by_EC_but_find_min_on_PI(self, target_layer: MoeHandle, gate_activations: torch.Tensor, **kwargs):
        '''history will be inverted (pi and ec)'''
        clusters, history = self._create_clusters_by_two_metric_by_clustering_algo(
            ExpertMatcher.find_by_dividive_clustering_PI_and_max_EC,
            ExpertLoggerDataConverter.compute_EC_matrix,
            ExpertLoggerDataConverter.compute_PI_matrix,
            target_layer, 
            gate_activations, 
            maximize_metric=True
            )
        return clusters, history
    
    def create_clusters_by_PI_objective_and_EC_product(self, target_layer: MoeHandle, gate_activations: torch.Tensor, **kwargs):
        '''Gets best clustering of dividive clustering watching on EC * (1/PI) product'''
        clusters, history = self._create_clusters_by_two_metric_by_clustering_algo(
            ExpertMatcher.find_by_dividive_clustering_max_PI_and_EC_product,
            ExpertLoggerDataConverter.compute_PI_matrix,
            ExpertLoggerDataConverter.compute_EC_matrix, 
            target_layer, 
            gate_activations, 
            maximize_metric=False
            )
        return clusters, history
    
    def create_clusters_by_EC_objective_and_PI_product(self, target_layer: MoeHandle, gate_activations: torch.Tensor, **kwargs):
        '''history will be inverted pi and ec'''
        clusters, history = self._create_clusters_by_two_metric_by_clustering_algo(
            ExpertMatcher.find_by_dividive_clustering_max_PI_and_EC_product,
            ExpertLoggerDataConverter.compute_EC_matrix,
            ExpertLoggerDataConverter.compute_PI_matrix,
            target_layer, 
            gate_activations, 
            maximize_metric=True
            )
        return clusters, history
    
    def create_clusters_by_PI_front_points(self, target_layer: MoeHandle, gate_activations: torch.Tensor, **kwargs):
        '''Returns many cases of best clusters depended on PI peak (front) points'''
        interesting_points, history = self._create_clusters_by_two_metric_by_clustering_algo(
            ExpertMatcher.find_by_dividive_clustering_objective_EC_front_of_PI,
            ExpertLoggerDataConverter.compute_PI_matrix,
            ExpertLoggerDataConverter.compute_EC_matrix, 
            target_layer, 
            gate_activations, 
            maximize_metric=True
            )
        
        
        return interesting_points, history #TODO auto best matched clustering_case
    
    def create_clusters_by_PI_greedy_aglomerative(self, target_layer: MoeHandle, gate_activations: torch.Tensor, log_history=False, **kwargs):
        '''Find clusters in one greedy step on min PI metric'''
        clusters, history = self._create_clusters_by_metric_by_clustering_algo(
            ExpertMatcher.find_by_aglomerative_clustering_greedy,
            ExpertLoggerDataConverter.compute_PI_matrix,
            target_layer, 
            gate_activations, 
            maximize_metric=False,
            log_history=log_history,
            )
        return clusters, history
    
    def create_clusters_by_EC_greedy_aglomerative(self, target_layer: MoeHandle, gate_activations: torch.Tensor, log_history=False, **kwargs):
        '''Find clusters in one greedy step on max EC metric. History (PI, EC) will be inverted'''
        clusters, history = self._create_clusters_by_metric_by_clustering_algo(
            ExpertMatcher.find_by_aglomerative_clustering_greedy,
            ExpertLoggerDataConverter.compute_EC_matrix,
            target_layer, 
            gate_activations, 
            maximize_metric=True,
            log_history=log_history,
            )
        return clusters, history
    
    def _create_clusters_by_two_metric_by_clustering_algo(self, 
                                                      clustering_algo: Callable, 
                                                      metric_matrix_algo_pi: Callable,
                                                      metric_matrix_algo_ec: Callable,
                                                      target_layer: MoeHandle, 
                                                      gate_activations: torch.Tensor, 
                                                      maximize_metric=False,
                                                      ) -> tuple[list[list[int]], list]:
        metric_matrix_pi = metric_matrix_algo_pi(self.model.num_experts, 
                                        gate_activations, 
                                        k = target_layer.moe.top_k)
        metric_matrix_ec = metric_matrix_algo_ec(self.model.num_experts, 
                                        gate_activations, 
                                        k = target_layer.moe.top_k)
        clusters, history = clustering_algo(metric_matrix_pi,
                                            metric_matrix_ec,
                                        maximize_metric=maximize_metric, 
                                        )
        return clusters, history
    
    def _create_clusters_by_metric_by_clustering_algo(self, 
                                                      clustering_algo: Callable, 
                                                      metric_matrix_algo: Callable, 
                                                      target_layer: MoeHandle, 
                                                      gate_activations: torch.Tensor, 
                                                      log_history=False, 
                                                      maximize_metric=True,
                                                      **kwargs) -> tuple[list[list[int]], list]:
        metric_matrix = metric_matrix_algo(self.model.num_experts, 
                                        gate_activations, 
                                        k = target_layer.moe.top_k)
        clusters, history = clustering_algo(metric_matrix, 
                                        maximize_metric=maximize_metric, 
                                        log_history=log_history,
                                        **kwargs)
        return clusters, history
        

    def _get_activations_from_gate(self, target_layers: list[MoeHandle]) -> dict[str, torch.Tensor]:
        name2activations = self.get_activations(target_layers, attach_to_gate_layer=True, attach_to_expert_layer=False)
        ExpertLoggerDataConverter.transform_activations_for_analysis(name2activations)
        return name2activations
    
    def _get_all_activations(self, target_layers: list[MoeHandle]) -> dict[str, torch.Tensor]:
        name2activations = self.get_activations(target_layers, attach_to_gate_layer=True, attach_to_expert_layer=True)
        ExpertLoggerDataConverter.transform_activations_for_analysis(name2activations)
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
                                    for handle in target_layers for expert in handle.experts]
            layer_names += experts_names
        return self._get_activations(layer_names)

    @torch.no_grad()
    def _get_activations(self, layer_names: list[str]) -> dict[str, list[torch.Tensor]]:
        experts_logger: ExpertsLogger = self.experts_logger(
            layer_of_interest=layer_names)
        experts_logger.reset()
        name2activation = experts_logger.get_activations(self.example_inputs)
        return name2activation
    
    def _parse_experts(self, name2activations, layer: int):
        pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)")
        result = {}
        for key in name2activations.keys():
            m = pattern.match(key)
            if m:
                layer_found = int(m.group(1))
                expert = int(m.group(2))
                if layer_found == layer:
                    result[expert] = name2activations[key]
        return result #dict[expert_id, tensor] 60 items for qwen

    def _get_tasks_expert_embeddings(self, name2activations, aggregation: Literal['mean', 'max']='mean', layers=None) -> pd.DataFrame:
        res = []
        layers = layers or [0]

        for layer in layers:
            experts = self._parse_experts(name2activations, layer)

            if aggregation == 'max':
                pass
                #embeddings = {e: activations[torch.argmax(torch.linalg.norm(activations, dim=-1))] for e, activations in experts.items()}
            elif aggregation == 'mean':
                for e, e_vals in experts.items():
                    for e_topic in e_vals:
                        embedding = torch.mean(e_topic, dim=0)
                        res.append(
                            (layer, e, *embedding.cpu().numpy())
                        )
            else:
                raise ValueError(f'Unknown aggregation: {aggregation}. Supported max, mean')
                    
        res = pd.DataFrame(res, columns=('layer', 'expert', *[f'emb_{i}' for i in range(len(res[-1]) - 2)]))
        return res 
