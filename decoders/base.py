from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .utils.stopping_condition import stopping_criterion, limit_past_key_values


default_generation_config = dict(
    temperature=0.2,
    top_k=50,
    top_p=1.0,
    do_sample=False,
    num_beams=1,
    use_cache=True,
    max_new_tokens=50,
)


def prepare_logits_processor(
        temperature=0.0, repetition_penalty=0.0, top_p=0.0, top_k=0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


class BaseDecoder:

    def __init__(
            self,
            tokenizer,
            model,
            gen_config=None,
            use_cache=True,
            device="cuda"
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.use_cache = use_cache
        self.device = device

        with torch.no_grad():
            self.model = model
            self.model.eval()

        self.max_length = min(self.tokenizer.model_max_length, 4096)

        # generation config
        self.generation_config = default_generation_config
        if gen_config is not None:
            self._update_generation_config(gen_config)

    def _update_generation_config(self, gen_config):
        for k, v in gen_config.items():
            if k in self.generation_config:
                self.generation_config[k] = v
            else:
                raise ValueError(f"Key '{k}' in 'gen_config' is not valid for 'default_generation_config'.")

    @abstractmethod
    def decode(self, input_ids, attention_mask, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def initialize(self, **kwargs):
        raise NotImplementedError

    def generate_logits_processor(self, generation_config):
        assert 0 < generation_config['top_p'] < 1, "top_p should between 0.0 and 1"
        if generation_config['do_sample']:
            logits_processor = prepare_logits_processor(
                temperature=generation_config['temperature'], 
                top_p=generation_config['top_p'], 
                top_k=generation_config['top_k']
            )
        else:
            logits_processor = LogitsProcessorList()
        return logits_processor

    @staticmethod
    def stopping_criterion(past_tensor, current_tensor, eos=None):
        return stopping_criterion(past_tensor, current_tensor, eos)

    @staticmethod
    def limit_past_key_values(past_key_values, limit):
        return limit_past_key_values(past_key_values, limit)
