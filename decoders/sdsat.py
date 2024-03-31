import torch
import time

from .base import BaseDecoder
from .model.kv_cache import initialize_past_key_values
from .model.utils import (
    generate_tree_buffers, 
    generate_draft_tree, 
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
    reset_past_key_values,
    tree_choice
)


class SDSATDecoderGreedy(BaseDecoder):

    def __init__(self, tokenizer, model, sats, **kwargs):
        super().__init__(tokenizer, model, **kwargs)

        self.name = "SDSAT_Greedy"
        self.stop_tokens = [self.tokenizer.eos_token_id]

        # semantic adatpive tokens
        self.sats = torch.tensor([int(i) for i in sats.split(",")], dtype=torch.int64).to(self.device)

    def limit_output(self, input_values, output_values):
        """Calculate the index of the first different element"""
        input_values = input_values[:, 1:]
        output_values = output_values[:, :-1]
        diff_tensor = torch.eq(input_values, output_values)
        diff_indices = torch.where(diff_tensor == False)
        if diff_indices[0].nelement() == 0:
            return input_values.shape[1]

        first_diff_index = diff_indices[1][0].item()
        return first_diff_index

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

    def _initialize_logits(self, model, input_ids, past_key_values):
        """
        Forward pass through the model to obtain the model outputs, and logits.


        Args:
        - model: The LLM for generation.
        - input_ids (torch.Tensor): The input tensor containing token ids.
        - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

        Returns:
        - logits (torch.Tensor): logits from the LLM.
        """
        outputs, logits = model(
            input_ids, past_key_values=past_key_values, output_orig=True
        )
        return logits

    @torch.no_grad()
    def decode(self, input_ids, attention_mask, nrof_sat, target_len):
        new_token = 0
        input_len = input_ids.shape[1]
        self.model.base_model.model.draft_mask = None
        logits = self._initialize_logits(self.model, input_ids, self.model.past_key_values)
        init_value = torch.argmax(logits, dim=-1)[:, -1]

        output_tensor = torch.tensor([init_value], device=self.device).unsqueeze(0)

        a_time = 0  # accept times of adaptive tokens
        c_time = 0  # generate times of adaptive tokens
        iter_times = 0  # iter times
        loop_times = []
        start_loop = time.perf_counter()
        while(True):
            blockt = torch.concat((init_value, self.sats[: nrof_sat]), dim=0).unsqueeze(0)

            # first infer
            t1 = time.perf_counter()
            output = self.model.base_model(
                blockt,
                use_cache=True,
                past_key_values=self.model.past_key_values
            )
            t2 = time.perf_counter()
            loop_times.append((t2 - t1) * 1000)  # ms

            logits = output.logits
            max_values1 = torch.argmax(logits, dim=-1)

            output_tensor = torch.concat((output_tensor, max_values1[:, :1]), dim=1)
            if self.early_stop(max_values1[:, 0]) != -1:
                break

            self.model.current_length_data.fill_(input_len + 1)

            # second infer
            blockt = max_values1
            output = self.model.base_model(
                blockt,
                use_cache=True,
                past_key_values=self.model.past_key_values
            )
            logits = output.logits
            max_values2 = torch.argmax(logits, dim=-1)
            indx = self.limit_output(blockt, max_values2)
            output_shift = 2 + indx
           
            self.model.current_length_data.fill_(input_len + output_shift)

            new_token += output_shift
            input_len += output_shift
            a_time += indx
            c_time += nrof_sat
            iter_times += 2
            init_value = max_values2[:, indx]

            stop_indx = self.early_stop(max_values2[:, :indx + 1])
            if stop_indx != -1:
                output_tensor = torch.concat((output_tensor, max_values2[:, :stop_indx + 1]), dim=1)
                break
            elif new_token >= target_len:
                output_tensor = torch.concat((output_tensor, max_values2[:, :indx + 1]), dim=1)[:, :target_len]
                break
            else:
                output_tensor = torch.concat((output_tensor, max_values2[:, :indx + 1]), dim=1)
        end_loop = time.perf_counter()

        if len(loop_times) > 0:
            avg_loop_time = sum(loop_times) / len(loop_times)
            token_per_s = new_token / (end_loop - start_loop)
            accept_rate = a_time / c_time if c_time !=0 else None
        else:
            avg_loop_time = None
            token_per_s = None
            accept_rate = None

        reset_past_key_values(self.model.past_key_values)
        return output_tensor, iter_times, accept_rate, avg_loop_time, token_per_s


class SDSATDecoderNucleus(BaseDecoder):

    def __init__(self, tokenizer, model, sats, **kwargs):
        super().__init__(tokenizer, model, **kwargs)

        self.name = "SDSAT_Nucleus"
        self.stop_tokens = [self.tokenizer.eos_token_id]

        self.sats = torch.tensor([int(i) for i in sats.split(",")], dtype=torch.int64).to(self.device)

    def initialize(self):
        self.logits_processor = self.generate_logits_processor(self.generation_config)

        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(self.model.base_model)
        self.model.past_key_values = past_key_values
        self.model.past_key_values_data = past_key_values_data
        self.model.current_length_data = current_length_data  # current length of past k,v of each layer

        self.model.current_length_data.zero_()  # this is for rerun
        self.model.base_model.model.draft_mask = None

        # Generate draft buffers
        self.draft_buffers = {}
        self.support_depth = []
        for depth, tree_structure in tree_choice.items():
            depth = int(depth.split('_')[-1])
            self.support_depth.append(depth) 
            self.draft_buffers[depth] = generate_tree_buffers(
                tree_structure, self.generation_config['top_k'], device = self.model.base_model.device
            )

    def early_stop(self, values):
        for i, _t in enumerate(self.stop_tokens):
            if _t in values:
                return i
        return -1

    def _initialize_logits(self, model, input_ids, past_key_values):
        """
        Forward pass through the model to obtain the model outputs, and logits.


        Args:
        - input_ids (torch.Tensor): The input tensor containing token ids.
        - model: The LLM for generation.
        - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

        Returns:
        - logits (torch.Tensor): logits from the LLM.
        """
        self.model.base_model.model.draft_mask = None  # draft_mask is not needed.
        outputs, logits = model(
            input_ids, past_key_values=past_key_values, output_orig=True
        )
        return logits

    @torch.no_grad()
    def decode(self, input_ids, attention_mask, nrof_sat, target_len):
        assert nrof_sat in self.support_depth, f"Not suppport tree_depth_{nrof_sat} yet, you can design the tree_depth_{nrof_sat} yourself."
        assert nrof_sat <= len(self.sats), f"The {nrof_sat} is larger than the number of {self.sats}, please check it."
        new_token = 0
        input_len = len(input_ids[0])
        logits = self._initialize_logits(self.model, input_ids, self.model.past_key_values)
        cur_length = input_len

        draft_buffer = self.draft_buffers[nrof_sat]
        iter_times = 0
        a_time = 0
        c_time = 0
        loop_times = []
        start_loop = time.perf_counter()
        for _ in range(2000):
            self.model.base_model.model.draft_mask = None  # draft mask is not needed in draft step 1.
            candidates, tree_candidates, input_ids = generate_draft_tree(
                self.model, input_ids, cur_length, logits, nrof_sat, self.sats,
                draft_buffer['tree_sampling_nums'], draft_buffer["retrieve_indices"],
                self.generation_config['top_k'], self.model.past_key_values, self.logits_processor, self.device
            )

            cur_length += 1
            new_token += 1
            self.model.base_model.model.draft_mask = draft_buffer["tree_attn_mask"]

            t1 = time.perf_counter()
            logits, outputs = tree_decoding(
                    self.model,
                    tree_candidates,
                    self.model.past_key_values,
                    draft_buffer["tree_position_ids"],
                    input_ids,
                    draft_buffer["retrieve_indices"],
                )
            t2 = time.perf_counter()
            loop_times.append((t2 - t1) * 1000)  # ms

            best_candidate, accept_length = evaluate_posterior(
                    logits,
                    candidates,
                    temperature=self.generation_config['temperature'], 
                    top_p=self.generation_config['top_p']
                )
            
            input_ids, logits, new_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    draft_buffer["retrieve_indices"],
                    outputs,
                    logits,
                    new_token,
                    self.model.past_key_values_data,
                    self.model.current_length_data,
                )

            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            a_time += accept_length.item()
            c_time += nrof_sat
            iter_times += 2
            if self.early_stop(input_ids[0, input_len:]) != -1 or new_token >= target_len:
                break

        end_loop = time.perf_counter()

        if new_token > 0:
            avg_loop_time = sum(loop_times) / len(loop_times)
            token_per_s = new_token.item() / (end_loop - start_loop)
            accept_rate = a_time / c_time if c_time !=0 else None
        else:
            avg_loop_time = None
            token_per_s = None
            accept_rate = None
        reset_past_key_values(self.model.past_key_values)
        return input_ids[:, input_len:], iter_times, accept_rate, avg_loop_time, token_per_s
    
            
