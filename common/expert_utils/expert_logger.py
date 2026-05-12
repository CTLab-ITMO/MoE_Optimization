from pathlib import Path 
from typing import * 
from itertools import chain
import torch

def is_gate(name: str, layer: Optional[torch.nn.Module] = None):
    return name.endswith('gate')

class ExpertsLogger():
    '''Connects to the model and write activations via torch hook'''
    
    def __init__(self, model: torch.nn.Module, layer_of_interest: Union[type, List[str]], gate_func: Callable = is_gate):
        self.model = model
        self.is_gate: Callable = gate_func 
        if isinstance(layer_of_interest, type):
            self.name_to_layer = {name: layer for name, layer in model.named_modules() if isinstance(layer, layer_of_interest) or gate_func(name, layer)}
        elif isinstance(layer_of_interest, (tuple, list)) and len(layer_of_interest) and isinstance(layer_of_interest[0], str):
            self.name_to_layer = {name: layer for name, layer in model.named_modules() if name in layer_of_interest}
        else: 
            raise TypeError('Unknown type of argument `layer_of_interest` passed: expected either type of expert layer or list of layers\' names (strings)')
        self.layer_to_name: Dict[torch.nn.Module, str] = {layer: name for name, layer in self.name_to_layer.items()}
        self.name_to_activation: Dict[str, list] = {name: [] for name in self.name_to_layer}
        self.hooks = []
        self.reset()

    @torch.no_grad()
    def get_activations(self, example_batches):
        for batch in example_batches:
            self.model(**batch)
        self.clean() #destroy old hooks from module, if we want to use it later independently from ExpertsLogger
        return self.name_to_activation

    def _gathering_hook(self):
        def hook_function(module, input, output):
            self.name_to_activation[
                self.layer_to_name[module]
            ].append(
                output
            )
            
        for layer in self.layer_to_name:
            self.hooks.append(
                layer.register_forward_hook(hook_function)
            )

    def reset(self):
        self.clean()
        self.current_forward = 0 
        self.name_to_activation = {name: [] for name in self.name_to_layer}
        self._gathering_hook()

    def clean(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []