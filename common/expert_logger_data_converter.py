from MoE_Optimization.common.expert_logger import is_gate
import torch


class ExpertLoggerDataConverter:
    
    @staticmethod  
    def _count_pair_occurrences(indices):
        """
        Count occurrences of each unordered index pair across all rows.

        Args:
            indices: torch.Tensor of shape (n_samples, k), dtype=torch.long,
                    each row contains k indices (e.g., from torch.topk).

        Returns:
            unique_pairs: torch.Tensor of shape (num_unique_pairs, 2), each row is a pair (i, j) with i < j.
            counts: torch.Tensor of shape (num_unique_pairs,), counts for each pair.
        """
        n, k = indices.shape
        device = indices.device

        # All unordered column index pairs (i, j) with i < j
        col_pairs = torch.combinations(torch.arange(k, device=device), r=2)  # (m, 2)

        # Gather actual index values for these column pairs across all rows
        all_pairs = indices[:, col_pairs]  # (n, m, 2)

        # Sort each pair so that the smaller index comes first (canonical order)
        sorted_pairs, _ = torch.sort(all_pairs, dim=-1)  # (n, m, 2)

        # Flatten to a list of all pairs
        flat_pairs = sorted_pairs.view(-1, 2)  # (n*m, 2)

        # Count unique pairs
        unique_pairs, counts = torch.unique(flat_pairs, dim=0, return_counts=True)

        return unique_pairs, counts
        
    @staticmethod
    def get_raw_cooc_counts(gate_indices: torch.Tensor, k=4) -> dict[tuple[int, int], int]:
        """Returns dict with raw co-occurence count. pair -> count"""
        G_k = torch.topk(gate_indices, k=k).indices
        pairs, counts = ExpertLoggerDataConverter._count_pair_occurrences(G_k)
        return dict(zip((tuple(p) for p in pairs.tolist()), counts.tolist()))
    
    @staticmethod
    def PI_score(expert1: int, expert2: int, cooc: dict, k: int=4) -> float:
        """
        Вычисляет PI-метрику между двумя экспертами.
        PI = количество их встреч парой / сумма встреч каждого из них с другими экспертами
        """
        
        pair_key = tuple(sorted((expert1, expert2)))
        cooc_count = cooc.get(pair_key, 0)
        
        # Считаем общее количество встреч для каждого эксперта
        total_expert1 = sum(count for (e1, e2), count in cooc.items() if e1 == expert1 or e2 == expert1)
        total_expert2 = sum(count for (e1, e2), count in cooc.items() if e1 == expert2 or e2 == expert2)
        
        pi_value = cooc_count / (total_expert1 + total_expert2 - 2 * cooc_count + 1) 
        return pi_value

    # exclusivity 
    @staticmethod
    def EC_score(expert1: int, expert2: int, cooc: dict, k: int=4) -> float:
        """
        Вычисляет EC-метрику для одного эксперта.
        EC = сумма встреч каждого эксперта с другими экспертами / количество их встреч парой
        """

        pair_key = tuple(sorted((expert1, expert2)))
        cooc_count = cooc.get(pair_key, 0)
        
        # Считаем общее количество встреч для каждого эксперта
        total_expert1 = sum(count for (e1, e2), count in cooc.items() if e1 == expert1 or e2 == expert1)
        total_expert2 = sum(count for (e1, e2), count in cooc.items() if e1 == expert2 or e2 == expert2)
        
        ec_value = (total_expert1 + total_expert2 - 2 * cooc_count) / (cooc_count + 1)  
        return ec_value
    
    @staticmethod
    def compute_EC_matrix(num_experts: int, gate: torch.Tensor, k: int=4) -> torch.Tensor:
        ec_matrix = torch.zeros((num_experts, num_experts))
        raw_cooc = ExpertLoggerDataConverter.get_raw_cooc_counts(gate, k)   
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                ec_value = ExpertLoggerDataConverter.EC_score(i, j, raw_cooc, k)
                ec_matrix[i, j] = ec_value
                ec_matrix[j, i] = ec_value  
        return ec_matrix

    @staticmethod
    def compute_PI_matrix(num_experts: int, gate: torch.Tensor, k: int=4) -> torch.Tensor:
        pi_matrix = torch.zeros((num_experts, num_experts))
        raw_cooc = ExpertLoggerDataConverter.get_raw_cooc_counts(gate, k)
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                pi_value = ExpertLoggerDataConverter.PI_score(i, j, raw_cooc, k)
                pi_matrix[i, j] = pi_value
                pi_matrix[j, i] = pi_value  
        return pi_matrix

    @staticmethod
    def _transform_activations_for_analysis(activations):
        '''Transform gate 3-tuples to just 1D vectors of expert's logits'''
        for layer_name in activations.keys():
            # if (is_gate(layer_name)): #DONT NEED IN OLD VERSION OF transformers, but need in new ones
            #     activations[layer_name] = [gate[0] for gate in activations[layer_name]] #get raw logits from tuple (logits, topk scroes, topk indices)
            if (is_gate(layer_name)): #we need activations without batches on topics/tasks. but for embeddings we need distinct batches for mean aggregation in future
                activations[layer_name] = torch.cat(activations[layer_name])