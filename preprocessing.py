import numpy as np
import h5py
import os
import torch
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Union
import numexpr as ne

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ridge_regression as sklearn_ridge_regression
import h5py
import os.path
import torch
from typing import Optional, Union, List

import numpy as np
from scipy.stats import pearsonr
from collections import Counter

import random
import re
from typing import List, Tuple


def get_ordered_representations(
    ordered_strings_to_find: list[str],
    prompt: str,
    tokenizer,
    model
):
    """
    Finds a specific sequence of string occurrences in a prompt and computes
    their averaged token-level representations for each model layer.

    The search is sequential: the n-th string is searched for only *after*
    the end of the (n-1)-th found string.

    Args:
        ordered_strings_to_find (list[str]): An ordered list of strings to
                                             find sequentially.
        prompt (str): The full text to process.
        tokenizer: A "fast" Hugging Face tokenizer.
        model: A Hugging Face model that returns hidden_states.

    Returns:
        tuple[list, list]:
        - A list of dictionaries for the successfully found sequential
          occurrences, e.g., [{'text': 'str', 'span': (start, end)}].
        - A list of tensors. Each tensor corresponds to a model layer and
          has shape [num_found_occurrences, hidden_size].
    """
    # 1. Perform a sequential search for the ordered strings.
    found_occurrences = []
    search_start_index = 0
    for s in ordered_strings_to_find:
        # Use str.find() with a start index to enforce sequential order.
        match_start = prompt.find(s, search_start_index)

        assert not match_start == -1, f"Error: Could not find '{s}' after character index {search_start_index}. Check the prompt and list of words."

        match_end = match_start + len(s)
        found_occurrences.append({"text": s, "span": (match_start, match_end)})
        # The next search must start after the end of the current match.
        search_start_index = match_end

    if not found_occurrences:
        return [], []

    # 2. Create a map from each character's position to its occurrence's index.
    char_to_occurrence_idx = [None] * len(prompt)
    for i, occurrence in enumerate(found_occurrences):
        start, end = occurrence["span"]
        for char_idx in range(start, end):
            char_to_occurrence_idx[char_idx] = i

    # 3. Tokenize and get hidden states from the model.
    inputs = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    inputs = inputs.to(model.device)
    offset_mapping = inputs.pop("offset_mapping").squeeze(0)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    # 4. Map each token back to its string occurrence.
    occurrence_to_tokens_map = [[] for _ in found_occurrences]
    for token_idx, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue

        occurrence_idx = char_to_occurrence_idx[end - 1]

        if occurrence_idx is not None:
            occurrence_to_tokens_map[occurrence_idx].append(token_idx)

    # 5. Average the representations for each occurrence, across all layers.
    all_layers_averaged_reps = []
    hidden_size = model.config.hidden_size

    for layer_states in hidden_states:
        layer_states = layer_states.squeeze(0)
        layer_occurrence_reps = []
        for token_indices in occurrence_to_tokens_map:
            if token_indices:
                word_vectors = layer_states[token_indices]
                mean_vector = word_vectors.mean(dim=0)
                layer_occurrence_reps.append(mean_vector)
            else:
                layer_occurrence_reps.append(torch.zeros(hidden_size))

        all_layers_averaged_reps.append(torch.stack(layer_occurrence_reps))

    return found_occurrences, all_layers_averaged_reps


def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i). A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.
    
    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the interpolation filter.
    
    [window] lobes of the sinc function will be used. [window] should be an integer.
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    print ("Doing lanczos interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))
    
    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)
    
    if rectify:
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)), 
                            np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        ## Construct new signal by multiplying the sinc matrix by the data ##
        newdata = np.dot(sincmat, data)

    return newdata

def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = t * cutoff
    pi = np.pi
    #val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val = ne.evaluate("window * sin(pi*t) * sin(pi*t/window) / (pi**2 * t**2)")
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0

    return val

def concat_past_features(X, N):
    """
    We want to concatenate the N previous hidden states of a model into its hidden state for each time-step
    """
    n_time, n_features = X.shape
    out = [X]
    for k in range(1, N + 1):
        # Pad the first k rows with zeros, then take the rest from X[:-k]
        shifted = torch.cat([torch.zeros(k, n_features, device=X.device, dtype=X.dtype), X[:-k]], dim=0)
        out.append(shifted)
    # Concatenate all along the feature axis
    return torch.cat(out, dim=1)

    # ensure contiguous memory before flattening
    return windows.contiguous().view(n_trs, (k + 1) * h)

def test_concat_prev_states():
    """
    Checks that `concat_prev_states` correctly appends the k previous
    hidden states (zero-padded when unavailable) along the feature
    dimension, without hard-coding any expected numbers.
    """
    k = 3
    n_trs, h = 5, 2
    x = torch.arange(1.0, n_trs * h + 1).reshape(n_trs, h)  # simple, verifiable data

    y = concat_past_features(x, N=k)

    # --- build expected output on the fly ---------------------------------
    zero = torch.zeros(h, dtype=x.dtype, device=x.device)

    expected_rows = []
    for t in range(n_trs):
        parts = [x[t]]  # current state
        # predecessors or zeros if out-of-range
        parts += [x[t - j] if t - j >= 0 else zero for j in range(1, k + 1)]
        expected_rows.append(torch.cat(parts, dim=0))

    expected = torch.stack(expected_rows, dim=0)

    # --- assertions -------------------------------------------------------
    assert y.shape == (n_trs, (k + 1) * h)
    assert torch.equal(y, expected), "Concatenated values are incorrect"
    print("✅  concat_prev_states works as expected.")
#test_concat_prev_states()

def delete_block_edges(arr, mask, n_begin, n_last, axis=1):
    """
    Deletes the first n_begin and the last n_last elements from each block in an array.
    Used to remove the first and last TRs of each experiment run.

    Args:
        arr (np.ndarray): The input array of shape (n_dim, len(mask)) or (len(mask), n_dim).
        mask (list or np.ndarray): A 1D array or list defining the blocks.
        n_begin (int): The number of elements to delete from the beginning of each block.
        n_last (int): The number of elements to delete from the end of each block.
        axis (int): The axis along which to slice the array (0 or 1).

    Returns:
        np.ndarray: The array with the specified elements removed.
    """
    mask = np.array(mask)
    unique_blocks = np.unique(mask)

    indices_to_keep = []

    for block in unique_blocks:
        # Find all indices for the current block
        block_indices = np.where(mask == block)[0]

        # Determine the slice for the middle part of the block
        # If n_last is 0, we take all elements until the end.
        end_index = -n_last if n_last > 0 else len(block_indices)
        middle_indices = block_indices[n_begin:end_index]

        indices_to_keep.extend(middle_indices)

    # Sort indices to maintain the original order of the elements
    indices_to_keep = np.sort(indices_to_keep)

    # Slice the array along the specified axis to keep only the desired elements
    if axis == 1:
        return arr[:, indices_to_keep]
    else:
        return arr[indices_to_keep, :]


def normalize_train_test(
    x: torch.Tensor, 
    block_ids: Optional[Union[torch.Tensor, List[int]]], 
    eps: float = 1e-8,
    time_dim = 0
) -> torch.Tensor:
    """
    Normalizes a tensor to have unit variance and zero mean per feature, 
    applied in groups belonging to the same data source (e.g., session, story, or subject).

    This is the PyTorch equivalent of the provided NumPy function.

    Args:
        x (torch.Tensor): The input data tensor of shape (n_features, n_timepoints).
        block_ids (Optional[torch.Tensor or List]): A 1D tensor or list of length 
            n_timepoints indicating the group membership for each timepoint.
            If None, all data is treated as a single block.
        eps (float): A small value added to the denominator for numerical stability 
            to avoid division by zero.
        time_dim (int): The dimension along which to compute the mean and std.

    Returns:
        torch.Tensor: The normalized data tensor with the same shape as x.
    """
    # 1. Input Handling and Validation
    # Ensure x is a tensor.
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # If block_ids are not provided, treat all data as one group.
    if block_ids is None:
        block_ids = torch.zeros(x.shape[-1], dtype=torch.long, device=x.device)
    # Ensure block_ids is a tensor on the same device as x.
    elif not isinstance(block_ids, torch.Tensor):
        block_ids = torch.tensor(block_ids, dtype=torch.long, device=x.device)
    elif block_ids.device != x.device:
        block_ids = block_ids.to(x.device)

    # 2. Initialization
    # Create an empty tensor to store the results, matching x's properties.
    normalized_x = torch.zeros_like(x)

    # 3. Group-wise Normalization
    # Find the unique block identifiers.
    unique_blocks = torch.unique(block_ids)
    mean_blocks = []
    std_blocks = []
    for block_id in unique_blocks:
        # Create a boolean mask for the current block.
        indices = (block_ids == block_id)
        
        # Select data for the current block using the mask.
        block_data = x[:, indices]
        
        # Calculate mean and std for the block.
        # dim=1 corresponds to the 'timepoint' or 'context' axis.
        mean = torch.mean(block_data, dim=time_dim, keepdim=True)
        mean_blocks.append(mean)
        std = torch.std(block_data, dim=time_dim, keepdim=True)
        std_blocks.append(std)
        # Normalize the block and store it in the output tensor.
        # The mask 'indices' ensures we place the results in the correct columns.
        normalized_x[:, indices] = (block_data - mean) / (std + eps)
    if len(unique_blocks) == 1:
        return normalized_x, mean, std

    else:
        return normalized_x, mean_blocks, std_blocks


import random
from typing import List, Tuple

def shuffle_words(words: List[str], percentage: float, seed: int = 42, exclude_tokens: List[str] = ["\n\n"]) -> Tuple[List[str], List[int]]:

    assert 0 <= percentage <= 1, "percentage must be between 0 and 1"

    # Identify positions of words to consider for shuffling
    shuffle_indices = [i for i, w in enumerate(words) if w not in exclude_tokens]
    num_to_shuffle = int(len(shuffle_indices) * percentage)

    if num_to_shuffle == 0:
        return words.copy(), list(range(len(words)))

    # Randomly sample indices to shuffle
    random.seed(seed)
    indices_to_shuffle = random.sample(shuffle_indices, num_to_shuffle)

    # Extract and shuffle the selected words
    words_to_shuffle = [words[i] for i in indices_to_shuffle]
    random.shuffle(words_to_shuffle)

    # Insert shuffled words back
    shuffled_words = words.copy()
    for idx, new_word in zip(indices_to_shuffle, words_to_shuffle):
        shuffled_words[idx] = new_word

    # Build inverse mapping (from new positions back to original indices)
    inverse_indices = list(range(len(words)))
    for i, new_word in zip(indices_to_shuffle, words_to_shuffle):
        original_index = words.index(new_word)  # Caution: If duplicates exist, this gives first match
        inverse_indices[i] = original_index

    return shuffled_words, inverse_indices

