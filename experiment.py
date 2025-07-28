from encoding import *
from preprocessing import get_ordered_representations, normalize_train_test, concat_past_features, lanczosinterp2D, delete_block_edges
from encoding import nested_blocked_cv, ridge_regression_fit_sklearn, ridge_regression_predict_torch
from analysis import pearson_correlation
import h5py
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 0. load fmri data and annotation
data_path_prefix = "./data/HP_data/fMRI"
HF_home = "/SWS/llms/nobackup/"
results_path_prefix = "./results/original_code"

fmri_data = np.load(f"{data_path_prefix}/data_subject_I.npy", allow_pickle=True) # raw fmri data for one subject
fmri_time = np.load(f"{data_path_prefix}/time_fmri.npy", allow_pickle=True) # timing of each fmri TRs in seconds
fmri_runs = np.load(f"{data_path_prefix}/runs_fmri.npy", allow_pickle=True) # indices of which TRs belong to which run.
stimuli_words = np.load(f"{data_path_prefix}/words_fmri.npy", allow_pickle=True) # list of words shown as stimuli (sequentially on a screen word by word)
word_times = np.load(f"{data_path_prefix}/time_words_fmri.npy", allow_pickle=True) # timing in seconds of the words

# 1. prepare the input sequence for the LLM (e.g. by changing formatting). Here we change "+" to "\n\n" and "@" to nothing (used to highlight italics)
LLM_input_sequence = []
for word in stimuli_words:
    LLM_input_sequence.append(word) if word != "+" and not "@" in word \
    else LLM_input_sequence.append(word.replace("+", "\n\n").replace("@",""))
assert len(word_times.tolist()) == len(LLM_input_sequence), "different length" # make sure we still have the same length as we have word timings.

# 2. compute LLM representations for these words.
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=HF_home)
model = AutoModel.from_pretrained(model_name, cache_dir=HF_home).to(device)
print("Loaded model and tokenizer.")

strings_to_find = LLM_input_sequence # we search for these words in sequential order in the full prompt to obtain avg representations for these.
prompt_text = " ".join(LLM_input_sequence)

# compute representations 
occurrences, representations = get_ordered_representations(
    ordered_strings_to_find=strings_to_find,
    prompt=prompt_text,
    tokenizer=tokenizer,
    model=model
)
print("Computed word level representations for the input sequence.")

# 3. Map from word-level LLM representations to TR level representations via Lanczos resampling
layer_idx = -7 
interpolated_representations = lanczosinterp2D(
    representations[layer_idx].to("cpu"),             # shape: (n_samples_input, n_dim)
    oldtime=word_times,      # shape: (n_samples_input,)
    newtime=fmri_time,     # shape: (n_samples_target,)
    window=3,         # (optional) number of lobes for the window
    cutoff_mult=1.0,  # (optional) cutoff frequency multiplier
    rectify=False     # (optional) rectification
)
print("Interpolated LLM representations to match fMRI TRs via Lanczos downsampling.")

# 4. Concatenate last x TRs of the LLM representation. No need to filter the first x features because we skip those anyway in the next step.
N_lag = 4  # number of previous TRs to concatenate
interpolated_representations = concat_past_features(torch.from_numpy(interpolated_representations), N_lag).numpy()

# 4.5 (optional): apply PCA or other dimensionality reduction to the LLM representations


# 5. Filter out TRs at the boundary of multiple runs

# Delete the first 2 and last 1 elements from each experiment run
n_begin = 20
n_last = 15
mask = fmri_runs # This is an integer mask indicating which TRs belong to which run.
final_representations = delete_block_edges(interpolated_representations, 
                                           mask, 
                                           n_begin, 
                                           n_last, 
                                           axis=0) # axis 0: time-dimension is the first in our data
print(f"Deleted edges from the LLM representations to match the filtering of fMRI runs (removed first {n_begin} and last {n_last} of every run).")

# 6. Run the nested blocked cross-validation to find the best alpha per voxel for ridge regression.

# Convert data to torch tensors and move to GPU
X = torch.tensor(final_representations, dtype=torch.float32, device=device) # shape (n_TRs, n_features)
y = torch.tensor(fmri_data, dtype=torch.float32, device=device) # shape (n_TRs, n_voxels)

# Set parameters for crossvalidation
split_function = "blocked" # divide data into uniform folds
block_labels = None  # Not needed for blocked splitting - useful for experiment-wise CV folds
n_splits_outer = 4   # Four blocks for outer CV (must be larger than 1)
n_splits_inner = 3   # Three blocks for inner CV (must be larger than 1)
gap = 15 # number of TRs to skip/discard in between train and test to avoid leakage
alphas = [0.1,1,10,100,1000,10000] # default vals from https://github.com/mtoneva/brain_language_nlp/blob/master/utils/utils.py 

print("Starting nested blocked cross-validation to find the best alpha per voxel for ridge regression")
# Obtain the best ridge regression penalties for each voxel independently (~25k alphas)
best_alphas, outer_scores = nested_blocked_cv(
    X[:880], y[:880],
    split_function=split_function,
    block_labels=block_labels,
    n_splits_outer=n_splits_outer,
    n_splits_inner=n_splits_inner,
    gap=gap,
    alphas=alphas,
    device=device
)
print("Best alphas found:", best_alphas)


block_labels = [[0 for _ in range(X.shape[0])]] # everything comes from the same story so we assume identical block_labels.

# 7. Fit the final encoding model on the first 911 TRs and evaluate on the remaining 300.
# for the last 300 TRs, we use the best alpha found in the nested CV and fit a final model to the normalized data. We then save the results to a h5 file
X_train_normalized, X_means, X_stds = normalize_train_test(X[:-300], block_ids=None)
y_train_normalized, y_means, y_stds = normalize_train_test(y[:-300], block_ids=None)
X_test_normalized = (X[-300:] - X_means) / (X_stds + 1e-8)
y_test_normalized = (y[-300:] - y_means) / (y_stds + 1e-8)

best_alphas = torch.from_numpy(np.array(best_alphas))
w_final = ridge_regression_fit_sklearn(X_train_normalized, y_train_normalized, best_alphas, device=device)

# 8. Use the coefficients of the final model to predict the (normalized) fMRI data and compute pearson's r and associated p-values per voxel.
y_pred_final = ridge_regression_predict_torch(X_test_normalized, w_final)

correlations, p_values = pearson_correlation(y_test_normalized, y_pred_final)
print("correlations mean", correlations.mean())
print("Number of significant voxels: ", len([v for v in p_values.cpu().numpy().tolist() if v <0.01]))

# 9. Save results into h5. 
# Why h5? We do not need to load everything at once, and it allows to have meta-data and structure without using pickle.
# check if h5 file already exists
results_filename = f'{results_path_prefix}/results_encoding_model.h5'
representation_name = f"layer {layer_idx}"
write_mode = "a" if os.path.isfile(results_filename) else "w" # append if file already exists

with h5py.File(results_filename, write_mode) as f:
    # add meta-data as attributes to the h5 file
    f.attrs["model_name"] = "Llama-3.2-1B-Full-Chapter" # replace with metadata for the experiment run
    # allow overwriting of results if they already existed
    if representation_name in f:
        print(f"overwriting group {representation_name}")
        del f[representation_name]
        
    group = f.create_group(representation_name)
    group.create_dataset('predictions', data=y_pred_final.cpu().numpy()) # save results as datasets
    group.create_dataset('ground_truth', data=y_test_normalized.cpu().numpy()) # save ground truth as datasets    
    group.create_dataset('correlations', data=correlations.cpu().numpy()) # save correlations per voxel
    group.create_dataset('p_values', data=p_values.cpu().numpy()) # save p value per voxel
    group.create_dataset("coefficients", data= w_final.cpu().numpy()) # save coefficients
    group.create_dataset("alphas", data=np.array(best_alphas))
print("DONE")