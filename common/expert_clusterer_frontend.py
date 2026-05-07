from MoE_Optimization.common.clustering_utils import SELECTED_TASKS, convert_inputs_to_tokens
from MoE_Optimization.common.expert_clusterer import ExpertClusterer
from MoE_Optimization.common.clustering_utils import MMLUBatchGenerator

one_layer_clustering_method = [
    'create_clusters_by_PI',
    'create_clusters_by_EC',
    'create_clusters_by_PI_but_find_max_on_EC',
    'create_clusters_by_EC_but_find_min_on_PI',
    'create_clusters_by_PI_objective_and_EC_product',
    'create_clusters_by_EC_objective_and_PI_product',
    'create_clusters_by_PI_greedy_aglomerative',
    'create_clusters_by_EC_greedy_aglomerative',
    'create_clusters_by_PI_front_points_and_Calinski_metric',
]

class ExpertClustererFrontend:
    def __init__(self, model, tokenizer):
        gen = MMLUBatchGenerator(subjects=SELECTED_TASKS, split='validation', batch_size=16, include_metadata=False)
        example_inputs = convert_inputs_to_tokens(gen, tokenizer, model.device)
        self.clusterer = ExpertClusterer(model, example_inputs)
    
    def match_experts(self, target_layers: list, clustering_method: str):
        match clustering_method:
            case 'create_clusters_by_PI':
                return self.clusterer.create_clusters_by_PI(target_layers)
            case 'create_clusters_by_EC':
                return self.clusterer.create_clusters_by_EC(target_layers)
            case 'create_clusters_by_PI_but_find_max_on_EC':
                return self.clusterer._create_clusters_by(
                    self.clusterer.create_clusters_by_PI_but_find_max_on_EC,
                    target_layers
                )
            case 'create_clusters_by_EC_but_find_min_on_PI':
                return self.clusterer._create_clusters_by(
                    self.clusterer.create_clusters_by_EC_but_find_min_on_PI,
                    target_layers
                )
            case 'create_clusters_by_PI_objective_and_EC_product':
                return self.clusterer._create_clusters_by(
                    self.clusterer.create_clusters_by_PI_objective_and_EC_product,
                    target_layers
                )
            case 'create_clusters_by_EC_objective_and_PI_product':
                return self.clusterer._create_clusters_by(
                    self.clusterer.create_clusters_by_EC_objective_and_PI_product,
                    target_layers
                )
            case 'create_clusters_by_PI_greedy_aglomerative':
                return self.clusterer._create_clusters_by(
                    self.clusterer.create_clusters_by_PI_greedy_aglomerative,
                    target_layers
                )
            case 'create_clusters_by_EC_greedy_aglomerative':
                return self.clusterer._create_clusters_by(
                    self.clusterer.create_clusters_by_EC_greedy_aglomerative,
                    target_layers
                )
            case 'create_clusters_by_PI_front_points_and_Calinski_metric':
                return self.clusterer.create_clusters_by_front_PI_and_max_Calinski(target_layers)
            case _:
                raise ValueError(f"Unknown clustering method: {clustering_method}")
                