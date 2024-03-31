import torch
from transformers import StoppingCriteria, StoppingCriteriaList


def limit_past_key_values(past_key_values, limit):
    """choose past key-values up to the limit."""
    new_list = []
    for elem in past_key_values:
        new_elem = list(elem)
        new_elem[0] = elem[0][:, :, :limit, :]
        new_elem[1] = elem[1][:, :, :limit, :]
        new_list.append(tuple(new_elem))
    return tuple(new_list)


def stopping_criterion(past_tensor, current_tensor, eos=None):
    assert past_tensor.shape == current_tensor.shape
    if torch.equal(past_tensor, current_tensor):
        tensor = current_tensor
        if eos is not None:
            if eos in current_tensor[0]:
                pos = (current_tensor[0] == eos).nonzero(as_tuple=True)[0]
                if pos.shape[0] > 1:
                    pos = pos[0].item()
                else:
                    pos = pos.item()
                return True, tensor, pos
            else:
                return True, tensor, -1
        return True, tensor
    else:
        if eos is not None:
            return False, current_tensor, False
        else:
            return False, current_tensor


def check_stop_cond(tensor, eos):
    if eos in tensor[0]:
        pos = (tensor[0] == eos).nonzero(as_tuple=True)[0]
        if pos.shape[0] > 1:
            pos = pos[0].item()
        else:
            pos = pos.item()
        return pos
    else:
        return -1


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)