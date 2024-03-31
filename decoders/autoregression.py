import torch
import time

from transformers import StoppingCriteria, StoppingCriteriaList

from .model.kv_cache import initialize_past_key_values
from .model.utils import top_p_filtering, reset_past_key_values
from .base import BaseDecoder


class ARDecoder(BaseDecoder):
    def __init__(self, tokenizer, model, **kwargs):
        super().__init__(tokenizer, model, **kwargs)

        self.name = "autoregression"
        self.stop_tokens = [self.tokenizer.eos_token_id]

    def early_stop(self, values):
        for i, _t in enumerate(self.stop_tokens):
            if _t in values:
                return i
        return -1

    def initialize(self):
        self.logits_processor = self.generate_logits_processor(self.generation_config)

        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(self.model.base_model)
        self.model.past_key_values = past_key_values
        self.model.past_key_values_data = past_key_values_data
        self.model.current_length_data = current_length_data

        self.model.current_length_data.zero_()
        self.model.base_model.model.draft_mask = None

    @torch.no_grad()
    def decode(self, input_ids, attention_mask, nrof_sat):
        new_token = 0
        input_len = input_ids.shape[1]
        self.model.base_model.model.draft_mask = None
        outputs = self.model.base_model(input_ids, past_key_values=self.model.past_key_values, use_cache=True)

        iter_times = 0
        loop_times = []
        start_loop = time.perf_counter()
        for _ in range(2000):
            if self.generation_config['do_sample']:
                assert 0 < self.generation_config['top_p'] < 1, "top_p should between 0.0 and 1"
                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = self.logits_processor(None, next_token_logits)
                input_id = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
                input_id = input_id.view(input_id.shape[0], 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            t1 = time.perf_counter()
            outputs = self.model.base_model(input_id, use_cache=True, past_key_values=self.model.past_key_values)
            t2 = time.perf_counter()
            loop_times.append((t2 - t1) * 1000)

            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1
            if self.early_stop(input_ids[0, input_len:]) != -1 or new_token > self.generation_config['max_new_tokens']:
                break

        end_loop = time.perf_counter()

        if len(loop_times) > 0:
            avg_loop_time = sum(loop_times) / len(loop_times)
            token_per_s = new_token / (end_loop - start_loop)
        else:
            avg_loop_time = None
            token_per_s = None

        reset_past_key_values(self.model.past_key_values)
        return input_ids[:, input_len:], new_token, avg_loop_time, token_per_s



