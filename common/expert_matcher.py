import MoE_Optimization.common.clustering_utils as clustering_utils

class ExpertMatcher:
        
    @staticmethod
    def find_by_dividive_clustering(adjacency_matrix_of_experts, maximize_metric=False, log_history=False) -> tuple[list[list[int]], list]:
        '''Returns experts pairs or triplets after dividive clustering and history'''
        clusters, history = clustering_utils._make_iterative_clustering(
            adjacency_matrix_of_experts,
            min_cluster_size=2,
            maximize_metric=maximize_metric,
            devide_worst_candidate_first=True,
            other_matrices_for_stats=None,
            log_history=log_history
            )
        return clusters, history
    
    
    #...There could be different strategies of finding experts for merge
    #Using different clustering approaches and algorithms