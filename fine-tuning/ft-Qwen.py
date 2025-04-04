import os
from typing import Optional

import safetensors.torch
import torch
from transformers import Trainer

from swift.plugin import Tuner, extra_tuners, optimizers_map
from swift.tuners import LoraConfig, Swift


class CustomTuner(Tuner):
    """LoRA training for LLM"""

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        return Swift.from_pretrained(model, model_id, **kwargs)

    @staticmethod
    def save_pretrained(
            model: torch.nn.Module,
            save_directory: str,
            state_dict: Optional[dict] = None,
            safe_serialization: bool = True,
            **kwargs,
    ) -> None:
        if state_dict is None:
            state_dict = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        model.save_pretrained(save_directory, state_dict=state_dict, safe_serialization=safe_serialization, **kwargs)

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        target_regex = r'^model.layers.*'
        lora_config = LoraConfig(
            task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=target_regex)
        return Swift.prepare_model(model, lora_config)


def create_custom_optimizer(args, model, dataset):

    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    llm_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in llm_parameters if n in decay_parameters],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in llm_parameters if n not in decay_parameters],
            'weight_decay': 0.0,
        },
    ]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


extra_tuners['custom'] = CustomTuner
optimizers_map['custom'] = create_custom_optimizer
