import MoE_Optimization.common.clustering_utils as clustering_utils
from typing import Optional
import numpy as np

class ExpertMatcher:
        
    @staticmethod
    def find_by_dividive_clustering(adjacency_matrix_of_experts, 
                                    maximize_metric=False, 
                                    log_history=False,
                                    other_matrices_for_stats: Optional[dict[str, np.ndarray]] = None) -> tuple[list[list[int]], list]:
        '''Returns experts pairs or triplets after dividive clustering and history'''
        clusters, history = clustering_utils._make_iterative_clustering(
            adjacency_matrix_of_experts,
            min_cluster_size=2,
            maximize_metric=maximize_metric,
            devide_worst_candidate_first=True,
            other_matrices_for_stats=other_matrices_for_stats,
            log_history=log_history
            )
        return clusters, history
    
    @staticmethod
    def find_by_dividive_clustering_PI_and_max_EC(PI_matrix, 
                                    EC_matrix: np.ndarray,
                                    maximize_metric=False,
                                    other_matrices_for_stats: dict[str, np.ndarray] = {}) -> tuple[list[list[int]], list]:
        '''Returns experts pairs or triplets after dividive clustering and history.
        Automatically finds num of clusters, where EC max.
        
        If change PI and EC matrices, and set maximize_metric=True, than it will work in the same way, but for EC as objective
        '''
        EC_metric_key = 'EC'
        clusters, history = clustering_utils._make_iterative_clustering(
            PI_matrix,
            min_cluster_size=2,
            maximize_metric=maximize_metric,
            devide_worst_candidate_first=True,
            other_matrices_for_stats= {EC_metric_key: EC_matrix} | other_matrices_for_stats,
            log_history=True
            )
        
        ec_value = 0
        ec_index = -1
        for i, step_dict in enumerate(history):
            if (maximize_metric==False):
                if step_dict[f'total_{EC_metric_key}_after_split'] > ec_value:
                    ec_value = step_dict[f'total_{EC_metric_key}_after_split']
                    ec_index = i
            else:
                if step_dict[f'total_{EC_metric_key}_after_split'] < ec_value:
                    ec_value = step_dict[f'total_{EC_metric_key}_after_split']
                    ec_index = i
        
        clusters = history[ec_index]["all_clusters"]
        return clusters, history
    
    @staticmethod
    def find_by_dividive_clustering_max_PI_and_EC_product(PI_matrix, 
                                    EC_matrix: np.ndarray,
                                    maximize_metric=False,
                                    other_matrices_for_stats: dict[str, np.ndarray] = {}) -> tuple[list[list[int]], list]:
        '''Returns experts pairs or triplets after dividive clustering and history.
        Automatically finds num of clusters, where EC * (1/PI) max.
        
        If change PI and EC matrices and set maximize_metric=True will be the same result, but for EC as objective.
        '''
        EC_metric_key = 'EC'
        clusters, history = clustering_utils._make_iterative_clustering(
            PI_matrix,
            min_cluster_size=2,
            maximize_metric=maximize_metric,
            devide_worst_candidate_first=True,
            other_matrices_for_stats= {EC_metric_key: EC_matrix} | other_matrices_for_stats,
            log_history=True
            )
        
        optimum_value = 0
        index_of_max_value = 0
        for i, step_dict in enumerate(history):
            #product of EC and 1/PI
            product = step_dict[f'total_{EC_metric_key}_after_split'] / step_dict['total_metric_after_split']
            if (maximize_metric==False):
                if product > optimum_value:
                    optimum_value = product
                    index_of_max_value = i
            else:
                if product < optimum_value:
                    optimum_value = product
                    index_of_max_value = i
        
        clusters = history[index_of_max_value]["all_clusters"]
        return clusters, history
    
    @staticmethod
    def find_by_dividive_clustering_objective_EC_front_of_PI(PI_matrix,
                                    EC_matrix,
                                    maximize_metric=True,
                                    other_matrices_for_stats: dict[str, np.ndarray] = {}) -> tuple[list[list[list[int]]], list]:
        '''Returns many clustering cases.
        Automatically finds clusters with peak PI values on pi front. Objective is EC, maximize_metric=True.
        
        Works bad if change EC_matrix and PI, and maximize_metric=False!
        '''
        PI_metric_key = 'PI'
        clusters, history = clustering_utils._make_iterative_clustering(
            EC_matrix,
            min_cluster_size=2,
            maximize_metric=maximize_metric,
            devide_worst_candidate_first=True,
            other_matrices_for_stats= {PI_metric_key: PI_matrix} | other_matrices_for_stats,
            log_history=True
            )
        
        pi_values = [step_dict[f'total_{PI_metric_key}_after_split'] for step_dict in history]
        front_points = clustering_utils._detect_pi_front_points(pi_values)
        interesting_steps = front_points['idx'].to_list()
        clustering_cases = []
        for i in interesting_steps:
            clustering_cases.append(history[i]['all_clusters'])
        clustering_cases.append(clusters) #last step also
        return clustering_cases, history
    
    @staticmethod
    def find_by_aglomerative_clustering_greedy(adjacency_matrix_of_experts, 
                                    maximize_metric=False, 
                                    log_history=False,
                                    other_matrices_for_stats: Optional[dict[str, np.ndarray]] = None) -> tuple[list[list[int]], list]:
        '''Returns pairs by creating them in order of min PI'''
        clusters, history = clustering_utils._make_greedy_agglomerative_pairs(
            adjacency_matrix_of_experts,
            maximize_metric=maximize_metric,
            other_matrices_for_stats=other_matrices_for_stats,
            log_history=log_history
            )
        return clusters, history
    
    #...There could be different strategies of finding experts for merge
    #Using different clustering approaches and algorithms