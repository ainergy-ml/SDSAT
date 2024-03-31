# adapted from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py

import numpy as np
import torch
import torch.nn.functional as F
import copy


# nodes: 4, 8, 8, 5, 3
tree_depth_5 = [[0],[1],[2],[3],
           [0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0],
            [0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,2,0],[0,2,1],[1,0,0],
            [0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,1,0],[0,0,1,1],
            [0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,2]]

# nodes: 4, 8, 8, 6, 5, 3, 2
tree_depth_7 = [[0],[1],[2],[3],
           [0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0],
            [0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,2,0],[0,2,1],[1,0,0],
            [0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,1,0],[0,0,1,1],[0,0,2,0],
            [0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,2],[0,0,0,1,0],[0,0,0,1,1],
            [0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,0,2],
            [0,0,0,0,0,0,0],[0,0,0,0,0,0,1]]

# nodes: 4, 8, 8, 6, 6, 5, 4, 3, 2
tree_depth_9 = [[0],[1],[2],[3],
           [0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0],
            [0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,2,0],[0,2,1],[1,0,0],
            [0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,1,0],[0,0,1,1],[0,0,2,0],
            [0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,2],[0,0,0,1,0],[0,0,0,1,1],[0,0,0,2,0],
            [0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,0,2],[0,0,0,0,1,0],[0,0,0,0,1,1],
            [0,0,0,0,0,0,0],[0,0,0,0,0,0,1],[0,0,0,0,0,0,2],[0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,2],
            [0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]]

# nodes: 4, 8, 8, 8, 6, 6, 6, 5, 4, 3, 2
tree_depth_11 = [[0],[1],[2],[3],
           [0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0],
            [0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,2,0],[0,2,1],[1,0,0],
            [0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,1,0],[0,0,1,1],[0,0,2,0],[0,0,2,1],[0,1,0,0],
            [0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,2],[0,0,0,1,0],[0,0,0,1,1],[0,0,0,2,0],
            [0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,0,2],[0,0,0,0,1,0],[0,0,0,0,1,1],[0,0,0,0,2,0],
            [0,0,0,0,0,0,0],[0,0,0,0,0,0,1],[0,0,0,0,0,0,2],[0,0,0,0,0,1,0],[0,0,0,0,0,1,1],[0,0,0,0,0,2,0],
            [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,2],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,2],[0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,2],
            [0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1]]

# nodes: 4, 8, 8, 8, 8, 8, 6, 6, 6, 5, 4, 3, 2
tree_depth_13 = [[0],[1],[2],[3],
           [0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0],
            [0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,2,0],[0,2,1],[1,0,0],
            [0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,1,0],[0,0,1,1],[0,0,2,0],[0,0,2,1],[0,1,0,0],
            [0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,2],[0,0,0,1,0],[0,0,0,1,1],[0,0,0,2,0],[0,0,0,2,1],[0,0,1,0,0],
            [0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,0,2],[0,0,0,0,1,0],[0,0,0,0,1,1],[0,0,0,0,2,0],[0,0,0,0,2,1],[0,0,0,1,0,0],
            [0,0,0,0,0,0,0],[0,0,0,0,0,0,1],[0,0,0,0,0,0,2],[0,0,0,0,0,1,0],[0,0,0,0,0,1,1],[0,0,0,0,0,2,0],
            [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,2],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,2,0],
            [0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,2],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,2,0],
            [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,2],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,2],[0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,2],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1]]

tree_choice = {
    "tree_depth_5": tree_depth_5,
    "tree_depth_7": tree_depth_7,
    "tree_depth_9": tree_depth_9,
    "tree_depth_11": tree_depth_11,
    "tree_depth_13": tree_depth_13,
}


def sample(logits, logits_processor, k=1, replacement=False):
    logits = logits_processor(None, logits)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    sampled_indices = torch.multinomial(probabilities, k, replacement=False)
    sampled_probs = torch.gather(probabilities, 1, sampled_indices)

    cumulative_sum = torch.cumsum(sampled_probs, dim=1)
    cumulative_sum = torch.cat(
        (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

    sampled_probs = sampled_probs / (1 - cumulative_sum)
    sampled_probs[torch.isinf(sampled_probs)] = -1
    sampled_probs[torch.isnan(sampled_probs)] = -1

    sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

    return sampled_indices, sampled_probs, probabilities

        
def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


class node:
    def __init__(self,parent=None,value=None,dict_key=None):
        self.parent=parent
        self.value=value
        if parent:
            self.depth=parent.depth+1
            parent.children.append(self)
        else:
            self.depth=0
        self.children=[]
        self.dict_key=dict_key
    def is_leaf(self):
        return len(self.children)==0

    def all_index(self):
        if not self.parent.parent:
            return [self.index]
        else:
            return self.parent.all_index()+[self.index]


class Tree:
    def __init__(self,tree_list):
        sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
        self.root=node()
        self.node_dic={}
        for tree_node in sorted_tree_list:
            cur_value=tree_node[-1]
            if len(tree_node)==1:
                cur_node=node(parent=self.root,value=cur_value,dict_key=tuple(tree_node))
            else:
                cur_parent=self.node_dic[tuple(tree_node[:-1])]
                cur_node = node(parent=cur_parent, value=cur_value,dict_key=tuple(tree_node))
            self.node_dic[tuple(tree_node)] = cur_node
        self.indexnode()

    def max_depth(self):
        return max([item.depth for item in self.node_dic.values()])

    def num_node_wchild(self):
        num_c=0
        for item in self.node_dic.values():
            if not item.is_leaf():
                num_c+=1
        return num_c

    def get_node_wchild(self):
        ns=[]
        for item in self.node_dic.values():
            if not item.is_leaf():
                ns.append(item)
        return ns

    def indexnode(self):
        cur_index=0
        for key in self.node_dic:
            cur_node=self.node_dic[key]
            if not cur_node.is_leaf():
                cur_node.index=cur_index
                cur_index+=1


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_tree_buffers(tree_choices, top_k, device="cuda"):
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # Sampling num of each node
    sampling_nums = [[1]]  # first node
    start = 0
    for i in range(len(depth_counts)):
        prev_parent_node = -1
        sam_num = 0
        sam_per_layer = []
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            if len(cur_tree_choice) == 1:
                sam_num += 1
                continue
            cur_parent_node = cur_tree_choice[-2]
            if cur_parent_node != prev_parent_node and prev_parent_node != -1:
                sam_per_layer.append(sam_num)
                sam_num = 1
            else:
                sam_num += 1
            prev_parent_node = cur_parent_node
        sam_per_layer.append(sam_num)
        start += depth_counts[i]
        sampling_nums.append(sam_per_layer)
        
    tree_attn_mask = torch.eye(tree_len, tree_len)
    tree_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # retrieve ancestor position
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
            tree_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    p_indices = [0 for _ in range(tree_len - 1)]
    b_indices = [[] for _ in range(tree_len - 1)]
    tree_indices[0] = 0
    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        inlayer_bias = 0
        b = []
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            cur_parent = cur_tree_choice[:-1]
            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    inlayer_bias += 1
                    parent = cur_parent
                    b = []
            else:
                parent = cur_parent
            tree_indices[start + j + 1] = cur_tree_choice[-1] + top_k * (i + bias) + 1
            p_indices[start + j] = inlayer_bias
            if len(b) > 0:
                b_indices[start + j] = copy.deepcopy(b)
            else:
                b_indices[start + j] = []
            b.append(cur_tree_choice[-1] + top_k * (i + bias) + 1)
        start += depth_counts[i]

    p_indices = [-1] + p_indices
    tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_tree_choices)):
        cur_tree_choice = sorted_tree_choices[-i - 1]
        retrieve_indice = []
        if cur_tree_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_tree_choice)):
                retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                retrieve_paths.append(cur_tree_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                 dim=1)

    p_indices = torch.tensor(p_indices)
    p_indices_new = p_indices[retrieve_indices]
    p_indices_new = p_indices_new.tolist()

    b_indices = [[]] + b_indices
    b_indices_new = []
    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = b_indices[index]
                if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        b_indices_new.append(iblist)

    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }
    tree_buffers["p_indices"] = p_indices_new
    tree_buffers["b_indices"] = b_indices_new
    tree_buffers["tree_sampling_nums"] = sampling_nums
    
    return tree_buffers


def generate_draft_tree(model, input_ids, input_len, logits, nrof_ctoken, ctokens,
               sampling_nums, retrieve_indices, top_k, past_key_values, logits_processor, device):
    """
    Generate draft tree (draft step 1). 
    The first token of the previous logits will be taken first.
    """
    # Get first input token_ids
    if logits_processor is not None:
        logit = logits[:, -1]
        logit = logits_processor(None, logit)
        probabilities = torch.nn.functional.softmax(logit, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(logits[:, -1])
        token = token[None, None]

    # Get token_ids of draft step 1   
    cur_input_ids = torch.cat((token.to(device), ctokens[None, :nrof_ctoken]), dim=1)
    tree_candidates = topK_generate(
        model, cur_input_ids, input_len, sampling_nums, top_k, past_key_values=past_key_values, logits_processor=logits_processor
    )

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates[retrieve_indices]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)

    # Update input_ids
    input_ids = torch.cat((input_ids, token.to(device)), dim=1)
    
    return cart_candidates, tree_candidates, input_ids


def topK_generate(model, input_ids, input_len, sampling_nums, top_k, past_key_values, logits_processor):
    """
    Generate tree candidates and obtain all candidates based on the design of the tree structure.
    """
    outputs, logits = model(input_ids, past_key_values=past_key_values, output_orig=True)

    # past_key_values = limit_past_key_values(past_key_values, input_len + 1)
    model.current_length_data.fill_(input_len + 1)

    tree_candidates = []
    for i in range(len(sampling_nums)):
        logit = logits[:, i]
        if logits_processor is not None:
            topk_index, topk_prob, op = sample(logit, logits_processor, k=top_k,)
        else:
            topk_index, topk_prob = torch.topk(logit, top_k, dim=-1).indices, torch.topk(logit, top_k, dim=-1).values
            op=None
        for j in sampling_nums[i]:
            select_ids = topk_index[:, :j]
            tree_candidates.append(select_ids)

    tree_candidates = torch.cat(tree_candidates, dim=-1).squeeze()

    # first token is verified, update the past_key_values
    # past_key_values = past_key_values
    return tree_candidates


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values

    
def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    draft_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - draft_position_ids (torch.Tensor): Positional IDs (Layer IDs in the Trie) of each draft token.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the draft position IDs to the length of the input sequence.
    position_ids = draft_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates. 
    # The model is expected to return each draft token's logits, and possibly other outputs.
    outputs, tree_logits = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    
    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]

    return logits, outputs


def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature

    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples * n_tokens, -1)

    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)


    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')

    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask


def evaluate_posterior(
    logits, candidates, temperature, top_p=0.8
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    elif top_p > 0:
        assert top_p < 1.0, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
    )
    # Update the past key values based on the selected tokens，本来past_key_values_data中就有这个值，为何这里还要再拷贝？
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = past_key_values_data[..., select_indices, :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Extract logits for the accepted tokens
    logits = logits[None, best_candidate, accept_length : accept_length + 1]

    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, new_token


def top_p_filtering(logits, top_p=0.0, filter_value=float('-inf')):
    # from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79


    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits