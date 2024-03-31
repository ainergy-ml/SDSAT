# adapted from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from transformers import AutoTokenizer
import os


class SDSATModel(nn.Module):

    def __init__(
        self,
        base_model,
        base_model_name_or_path,
        trust_remote_code,
    ):
        """
        Args:
            base_model (nn.Module): The LLM to be used.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, trust_remote_code=trust_remote_code)

    def get_tokenizer(self):

        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        base_model_path="codellama/CodeLlama-7b-instruct-hf",
        **kwargs,
    ):
        """
        Args:
            base_model_path (str): Name or path of the LLM to load.

        Returns:
            RestModel
        """
            
        base_model = KVLlamaForCausalLM.from_pretrained(
            base_model_path, **kwargs
        )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        model = cls(
            base_model,
            base_model_path,
            trust_remote_code,
        )

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        nrof_update_past_kv=None,
        output_orig=False,
        position_ids=None,
    ):
        """Forward pass of the LLM.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from the LM head.
        """
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                nrof_update_past_kv=nrof_update_past_kv,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])

        if output_orig:
            return outputs, orig
        raise NotImplementedError
