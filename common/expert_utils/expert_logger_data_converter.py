from MoE_Optimization.common.expert_utils.expert_logger import is_gate
import torch


class ExpertLoggerDataConverter:
    '''Create matrices and other data from raw outputs (activations and gate tensors)'''
    
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
    def get_raw_cooc_counts(gate_topk_indices_for_each_token: torch.Tensor) -> dict[tuple[int, int], int]:
        """Returns dict with raw co-occurence count. pair -> count"""
        pairs, counts = ExpertLoggerDataConverter._count_pair_occurrences(gate_topk_indices_for_each_token)
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
    def compute_EC_matrix(num_experts: int, gate_topk_indices_for_each_token: torch.Tensor, k: int=4) -> torch.Tensor:
        ec_matrix = torch.zeros((num_experts, num_experts))
        raw_cooc = ExpertLoggerDataConverter.get_raw_cooc_counts(gate_topk_indices_for_each_token)   
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                ec_value = ExpertLoggerDataConverter.EC_score(i, j, raw_cooc, k)
                ec_matrix[i, j] = ec_value
                ec_matrix[j, i] = ec_value  
        return ec_matrix

    @staticmethod
    def compute_PI_matrix(num_experts: int, gate_topk_indices_for_each_token: torch.Tensor, k: int=4) -> torch.Tensor:
        pi_matrix = torch.zeros((num_experts, num_experts))
        raw_cooc = ExpertLoggerDataConverter.get_raw_cooc_counts(gate_topk_indices_for_each_token)
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                pi_value = ExpertLoggerDataConverter.PI_score(i, j, raw_cooc, k)
                pi_matrix[i, j] = pi_value
                pi_matrix[j, i] = pi_value  
        return pi_matrix
    
    @staticmethod
    def _convert_coocurencies_matrix_to_dict(coocurencies_matrix: torch.Tensor):
        """Convert an upper-triangular co-occurrence matrix to a pair->value dict.

        The diagonal is ignored. Only entries with i < j are included.
        """
        if coocurencies_matrix.dim() != 2 or coocurencies_matrix.shape[0] != coocurencies_matrix.shape[1]:
            raise ValueError("coocurencies_matrix must be a square 2D tensor")

        num_experts = coocurencies_matrix.shape[0]
        cooc_dict: dict[tuple[int, int], float] = {}
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                cooc_dict[(i, j)] = coocurencies_matrix[i, j].item()
        return cooc_dict
    
    @staticmethod
    def compute_PI_matrix_based_on_cooc(num_experts: int, coocurencies_matrix: torch.Tensor, k: int=4) -> torch.Tensor:
        pi_matrix = torch.zeros((num_experts, num_experts))
        raw_cooc = ExpertLoggerDataConverter._convert_coocurencies_matrix_to_dict(coocurencies_matrix)
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                pi_value = ExpertLoggerDataConverter.PI_score(i, j, raw_cooc, k)
                pi_matrix[i, j] = pi_value
                pi_matrix[j, i] = pi_value  
        return pi_matrix
    
    @staticmethod
    def compute_EC_matrix_based_on_cooc(num_experts: int, coocurencies_matrix: torch.Tensor, k: int=4) -> torch.Tensor:
        ec_matrix = torch.zeros((num_experts, num_experts))
        raw_cooc = ExpertLoggerDataConverter._convert_coocurencies_matrix_to_dict(coocurencies_matrix)
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                ec_value = ExpertLoggerDataConverter.EC_score(i, j, raw_cooc, k)
                ec_matrix[i, j] = ec_value
                ec_matrix[j, i] = ec_value  
        return ec_matrix
    
    @staticmethod            
    def transform_gate_activations(activations: dict, top_k=4):
        '''Transform gate's n-tuples to just 1D vectors of top_k expert's indices.
        Note: in case of Qwen model - transform weight output vectors to expert indices.
        In case of DeepSeek model, find in output tuple int tensor and return it as indices.
        
        Changed shape in dict with `gate` word: Tensor(n_batches * len_in_batch, top_k_indices_of_experts)
        '''
        for layer_name in activations.keys():
        # if (is_gate(layer_name)): #DONT NEED IN OLD VERSION OF transformers, but need in new ones
        #     activations[layer_name] = [gate[0] for gate in activations[layer_name]] #get raw logits from tuple (logits, topk scroes, topk indices)
            if (is_gate(layer_name)): #we need activations without batches on topics/tasks. but for embeddings we need distinct batches for mean aggregation in future
                if (isinstance(activations[layer_name][0], tuple)):
                    probable_index_of_gates_idx = None #check index
                    for i, tuple_element in enumerate(activations[layer_name][0]): #heuristic for finding gate idx in case of tuple return
                        if not tuple_element.dtype.is_floating_point:
                            probable_index_of_gates_idx = i
                            break
                    if (probable_index_of_gates_idx is None):
                        raise ValueError(f"Expert indices not found in gate activations tuple. Probably the model not supported yet for that algo.")
                    activations[layer_name] = [gate_return[probable_index_of_gates_idx] 
                                                        for gate_return in activations[layer_name]]
                else:
                    if (activations[layer_name][0].dtype.is_floating_point):
                        activations[layer_name] = [torch.topk(gate_weights, k=top_k).indices for gate_weights in activations[layer_name]]
                
                activations[layer_name] = torch.cat(activations[layer_name])