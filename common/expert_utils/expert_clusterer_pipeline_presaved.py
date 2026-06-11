from typing import Iterable

import torch
from transformers import PreTrainedTokenizer
from typing import Optional

from MoE_Optimization.common.clustering_utils import SELECTED_TASKS
from MoE_Optimization.common.expert_utils.expert_clusterer import ExpertClusterer
from MoE_Optimization.common.input_data_utils.mmlu_batch_generator import MMLUBatchGenerator
from MoE_Optimization.common.input_data_utils.tokenization_utils import convert_text_inputs_to_tokens
from MoE_Optimization.common.moe_utils import MoeHandle

class ExpertClustererPipelinePresaved:
    '''
    The same as `ExpertClustererPipeline`, but don't rely on saved gate outputs, use them
    as `coocurencies_matrix`
    '''
    
    def __init__(self, model):
        '''
        Parameters:
            example_inputs: iterable of batches[N, seq_len, embedding_size]
            generator: generator that produce batches of text questions
        '''
        
        self.clusterer = ExpertClusterer(model, [])
            
    def match_experts_one_layer_saved_gate(self, target_layer: MoeHandle, clustering_method: str, coocurencies_matrix, max_cluster_size=None):
        '''
        
        Params:
            max_cluster_size: appliable only for algorithms that take last cluster solution from clustering history. 
            For solutions with find different metrics doesn't works.
        '''
        match clustering_method:
            case 'create_clusters_by_PI':
                #TODO refactor of using saved cooc_matrix
                return self.clusterer.create_clusters_by_PI_one_layer_based_on_freqencies(target_layer, coocurencies_matrix, max_cluster_size=max_cluster_size)
            case 'create_clusters_by_EC':
                return self.clusterer.create_clusters_by_EC_one_layer_based_on_freqencies(target_layer, coocurencies_matrix, max_cluster_size=max_cluster_size)
            case 'create_clusters_by_PI_greedy_aglomerative':
                return self.clusterer.create_clusters_by_PI_greedy_aglomerative_based_on_freq(target_layer, coocurencies_matrix, max_cluster_size=max_cluster_size)
            case 'create_clusters_by_EC_greedy_aglomerative':
                return self.clusterer.create_clusters_by_EC_greedy_aglomerative_based_on_freq(target_layer, coocurencies_matrix, max_cluster_size=max_cluster_size)
            case _:
                raise ValueError(f"Unknown clustering method or not implemented: {clustering_method}")
                