import torch
from datasets import load_dataset
from typing import Dict, List, Generator, Optional
import numpy as np

from transformers import BitsAndBytesConfig
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.sparse.csgraph import laplacian

SELECTED_TASKS = ["anatomy",
            # "conceptual_physics",
            # "human_sexuality",
            "machine_learning",
            "management",
            # "marketing",
            # "nutrition",
             "philosophy",
            # "us_foreign_policy",
              "world_religions"]


class MMLUBatchGenerator:
    """
    Generator that yields batches of 64 questions for each MMLU subject
    """

    def __init__(self,
                 subjects: Optional[List[str]] = None,
                 split: str = "test",
                 batch_size: int = 64,
                 shuffle: bool = False,
                 include_metadata: bool = True):
        """
        Initialize the MMLU batch generator

        Args:
            subjects: List of subject names (None = all 57 subjects)
            split: "test", "validation", or "dev"
            batch_size: Number of questions per batch (default: 64)
            shuffle: Whether to shuffle questions within each subject
            include_metadata: Whether to include subject name in output
        """
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.include_metadata = include_metadata

        # Load all subjects if none specified
        if subjects is None:
            # Get all available subjects
            self.subjects = self._get_all_subjects()
        else:
            self.subjects = subjects

        print(
            f"Initialized MMLU batch generator with {len(self.subjects)} subjects")
        print(f"Split: {split}, Batch size: {batch_size}")

    def _get_all_subjects(self) -> List[str]:
        """Get list of all 57 MMLU subjects"""
        # This is a comprehensive list of all MMLU subjects
        return [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios",
            "nutrition", "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology",
            "us_foreign_policy", "virology", "world_religions"
        ]

    def _format_question(self, example: Dict, subject: str) -> Dict:
        """Format a single question with its choices"""
        question = example['question']
        choices = example['choices']
        answer = example['answer']

        # Format choices as A, B, C, D
        formatted_choices = "\n".join(
            [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

        if self.include_metadata:
            return {
                'subject': subject,
                'question': question,
                'choices': choices,
                'formatted_question': f"{question}\n{formatted_choices}",
                'answer': answer,
                'answer_letter': chr(65 + answer) if answer < 4 else '?'
            }
        else:
            return question

    def __iter__(self) -> Generator:
        """
        Generator that yields batches of 64 questions for each subject
        """
        total_subjects = len(self.subjects)

        for subject_idx, subject in enumerate(self.subjects, 1):
            try:
                print(
                    f"\n📚 Loading subject {subject_idx}/{total_subjects}: {subject}")

                # Load the subject
                dataset = load_dataset(
                    "cais/mmlu", subject, trust_remote_code=True)
                split_data = dataset[self.split]

                # Get first 64 questions (or all if less than 64)
                num_questions = min(len(split_data), self.batch_size)
                indices = list(range(num_questions))

                if self.shuffle:
                    import random
                    random.shuffle(indices)

                # Create batch
                batch = []
                for idx in indices:
                    example = split_data[idx]
                    formatted = self._format_question(example, subject)
                    batch.append(formatted)

                print(f"  ✓ Created batch with {len(batch)} questions")

                # Yield the batch with metadata
                if self.include_metadata:
                    yield {
                        'subject': subject,
                        'batch_size': len(batch),
                        'total_questions': len(split_data),
                        'questions': batch
                    }
                else:
                    yield batch

            except Exception as e:
                print(f"  ❌ Error loading {subject}: {e}")
                continue

    def __len__(self) -> int:
        """Return number of subjects (number of batches)"""
        return len(self.subjects)

    def get_subject_batch(self, subject: str) -> Optional[Dict]:
        """Get a single batch for a specific subject"""
        for batch in self:
            if batch['subject'] == subject:
                return batch
        return None


def get_qwen(model=None, quantization_config=None, cache_dir=None):

    MODEL = "Qwen/Qwen1.5-MoE-A2.7B"

    if quantization_config is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_compute_dtype=torch.float16,  # Compute in float16 for speed
            bnb_4bit_use_double_quant=True,  # Double quantization for extra memory savings
            # Normalized float 4-bit (optimal for LLMs)
            bnb_4bit_quant_type="nf4"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True,
                                                device_map='cuda',
                                                )
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                                device_map='cuda',
                                                dtype=torch.float16,
                                                quantization_config=quantization_config,
                                                cache_dir=cache_dir
                                                )
    return model, tokenizer

def convert_inputs_to_tokens(gen: MMLUBatchGenerator, tokenizer, device):
   converted_batches = []
   for batch in gen:
        inputs = tokenizer(batch, return_tensors="pt",
                padding=True,  # Pad to longest in batch
                truncation=True,
                max_length=512,  # Adjust based on your needs
                add_special_tokens=True
        )
        converted_batches.append({k: v.to(device) for k, v in inputs.items()})
   return converted_batches


def spectral_bisect(mat, nodes, min_cluster_size=2):
    # make graph bisection using Laplase operator and Fielder vector
    # more weight on edges - more connectivity. Bisectrion tries to delete weak edges to maximize weight on parts.
    nodes = np.asarray(nodes)
    if len(nodes) < 2 * min_cluster_size:
        return nodes.tolist(), []

    sub = mat[np.ix_(nodes, nodes)].numpy()
    if np.allclose(sub, 0.0):
        order = np.arange(len(nodes))
    else:
        lap = laplacian(sub, normed=True)
        evals, evecs = np.linalg.eigh(lap)
        fiedler = evecs[:, 1] if evecs.shape[1] > 1 else evecs[:, 0]
        order = np.argsort(fiedler)

    split = len(nodes) // 2
    split = max(min_cluster_size, min(
        split, len(nodes) - min_cluster_size))
    left = nodes[order[:split]].tolist()
    right = nodes[order[split:]].tolist()

    if not left or not right:
        strengths = sub.sum(axis=1)
        order = np.argsort(strengths)
        left = nodes[order[:split]].tolist()
        right = nodes[order[split:]].tolist()

    return left, right

def _compute_metric_on_dense_cluster(mat, nodes, reduce="mean"):
    '''
    Args:
        nodes: indexes of nodes in cluster
    '''
    #if mean - have sence of density
    if mat is None:
        return np.nan
    nodes = np.asarray(nodes)
    if len(nodes) < 2:
        return 0.0

    sub = mat[np.ix_(nodes, nodes)]
    upper = np.triu(sub, 1)
    pair_count = len(nodes) * (len(nodes) - 1) // 2
    total = float(upper.sum())

    if reduce == "sum":
       return total
    if reduce == "mean":
        return total / pair_count if pair_count > 0 else 0.0

    raise ValueError("reduce must be 'mean' or 'sum'")
    
def _weighted_average_cluster_metric(mat, clusters):
    if mat is None:
        return np.nan

    total_sum = 0.0
    total_pairs = 0

    for cluster in clusters:
        nodes = np.asarray(cluster)
        if len(nodes) < 2:
            continue
        sub = mat[np.ix_(nodes, nodes)]
        upper = np.triu(sub, 1)
        total_sum += float(upper.sum())

    return total_sum / total_pairs if total_pairs > 0 else 0.0
    
def _make_iterative_clustering(adjacency_matrix_of_experts: np.ndarray, 
                              min_cluster_size=2,
                              maximize_metric=True, 
                              other_matrices_for_stats: Optional[dict[str, np.ndarray]] = None,
                              devide_worst_candidate_first = True,
                              log_history=False) -> tuple[list, list]:
    '''
    Makes iterative deviding clustering using `adjacency_matrix_of_experts`.
    Additionally compute stats of main metric on every step. If `other_matrices_for_stats` provided then
    ads their metrics (using name of key) during clustering in history.
        
    Returns: 
            clusters: as indexes
            history: as dicts
    '''

    if adjacency_matrix_of_experts.ndim != 2 or adjacency_matrix_of_experts.shape[0] != adjacency_matrix_of_experts.shape[1]:
        raise ValueError("adjacency_matrix_of_experts must be a square matrix")

    clusters = [list(range(adjacency_matrix_of_experts.shape[0]))]
    history = []
    step = 0

    while True:
        objective_mat = adjacency_matrix_of_experts
        if (maximize_metric == False):
            objective_mat = 1 / (objective_mat + 0.0001)
            
        candidates = [
            (_compute_metric_on_dense_cluster(objective_mat, cluster), len(cluster), idx)
            for idx, cluster in enumerate(clusters)
            if len(cluster) >= 2 * min_cluster_size
        ]
        if not candidates:
            break

        if (devide_worst_candidate_first):
            _, _, split_idx = max(candidates, key=lambda item: (-item[0], item[1]))
        else:
            _, _, split_idx = max(candidates, key=lambda item: (item[0], item[1]))
            
        cluster_to_divide = clusters.pop(split_idx)
        left, right = spectral_bisect(objective_mat, cluster_to_divide, min_cluster_size=min_cluster_size)
            
        if len(left) < min_cluster_size or len(right) < min_cluster_size:
            clusters.insert(split_idx, cluster_to_divide)
            break

        clusters.extend([left, right])
        step += 1
        
        if (log_history):
            step_dict = {
                    "step": step,
                    "devided_cluster": cluster_to_divide,
                    "left": left,
                    "right": right,
                    "metric_of_split_cluster": _compute_metric_on_dense_cluster(adjacency_matrix_of_experts, cluster_to_divide),
                    "total_metric_after_split": _weighted_average_cluster_metric(adjacency_matrix_of_experts, clusters),
                    }
            
            if (other_matrices_for_stats is not None):
                for key in other_matrices_for_stats.keys():
                    step_dict[f'{key}_of_split_cluster'] = _compute_metric_on_dense_cluster(other_matrices_for_stats[key], cluster_to_divide)
                    step_dict[f'total_{key}_after_split'] = _weighted_average_cluster_metric(other_matrices_for_stats[key], clusters)
                
            history.append(step_dict)

    return clusters, history 