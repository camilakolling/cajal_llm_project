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
from torch import Tensor
from tqdm import tqdm

from preprocessing import normalize_train_test
from analysis import pearson_correlation
# NestedCrossvalidation.py

# --- Blocked Time Series Splitter ---
def blocked_time_series_split(n_samples, n_splits=5, test_size=None, gap=0, device='cpu'):
    """
    Generate splits of different train size, with a gap between train and test.
    The test sets are always later in time than the train sets (no shuffling), and has the same size.
    The train sets are of differing size, which is how we get the folds.
    """
    if test_size is None:
        test_size = n_samples // (n_splits + 1)
    indices = torch.arange(n_samples, device=device)
    splits = []
    for i in range(n_splits):
        train_end = test_size * (i + 1) - gap
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        train_indices = indices[:train_end]
        test_indices = indices[test_start:test_end]
        splits.append((train_indices, test_indices))
    return splits

def sliding_window_split(n_samples, train_size, test_size, gap=0, step=1, device='cpu'):
    """
    Instead of using a growing training set across the folds, we use same-size sliding windows
    for the training splits.
    """
    splits = []
    max_start = n_samples - (train_size + gap + test_size) + 1
    for start in range(0, max_start, step):
        train_start = start
        train_end = train_start + train_size
        test_start = train_end + gap
        test_end = test_start + test_size
        train_indices = torch.arange(train_start, train_end, device=device)
        test_indices = torch.arange(test_start, test_end, device=device)
        splits.append((train_indices, test_indices))
    return splits

# 1. Sliding/Rolling Window Splitter

def sliding_window_split(n_samples, train_size, test_size, gap=0, step=1):
    """
    Returns a list of (train_indices, test_indices) for each sliding window.
    """
    splits = []
    max_start = n_samples - (train_size + gap + test_size) + 1
    for start in range(0, max_start, step):
        train_start = start
        train_end = train_start + train_size
        test_start = train_end + gap
        test_end = test_start + test_size
        train_indices = np.arange(train_start, train_end)
        test_indices = np.arange(test_start, test_end)
        splits.append((train_indices, test_indices))
    return splits

# 2. Blocked Cross-Validation Splitter
def blocked_cv_split(n_samples, n_blocks, gap=0):
    """
    Returns a list of (train_indices, test_indices) for blocked cross-validation.
    """
    block_size = n_samples // n_blocks
    indices = np.arange(n_samples)
    splits = []
    for i in range(n_blocks):
        test_start = i * block_size
        test_end = test_start + block_size
        train_indices = np.concatenate([
            indices[:max(0, test_start-gap)],
            indices[min(n_samples, test_end+gap):]
        ])
        test_indices = indices[test_start:test_end]
        splits.append((train_indices, test_indices))
    return splits

# 3. Expanding Window Splitter (TimeSeriesSplit)
def expanding_window_split(n_samples, n_splits, test_size, gap=0):
    """
    Returns a list of (train_indices, test_indices) with expanding train set.
    """
    indices = np.arange(n_samples)
    splits = []
    for i in range(n_splits):
        train_end = (i+1)*test_size - gap
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        train_indices = indices[:train_end]
        test_indices = indices[test_start:test_end]
        splits.append((train_indices, test_indices))
    return splits

# 4. Custom Block Splitter (sessions/subjects/experiments)
def custom_block_split(block_labels, n_splits=4, gap=0):
    """
    Returns a list of (train_indices, test_indices) for each unique block label.
    block_labels: 1D array-like, e.g. [0,0,0,1,1,2,2,2,2,...]

    How and when to use:
    This crossvalidation splitter makes sense when we have multiple subjects, multiple sessions, experiment blocks etc.

    We concatenate all of the data (after removing the first TRs), yielding a single sequence.
    We annotate this sequence with block_labels indicating which index they belong to.
    We can have a dictionary that maps these back to their meaning {0: "subj0_session1", ...}
    
    """
    block_labels = np.array(block_labels)
    unique_blocks = np.unique(block_labels)
    indices = np.arange(len(block_labels))
    splits = []
    counter=0
    for block in unique_blocks:
        test_mask = (block_labels == block)
        test_indices = indices[test_mask]
        before = max(0, test_indices[0] - gap)
        after = min(len(block_labels), test_indices[-1] + 1 + gap)
        train_mask = np.ones(len(block_labels), dtype=bool)
        train_mask[before:after] = False
        train_indices = indices[train_mask]
        splits.append((train_indices, test_indices))
        counter += 1
        if counter == n_splits:
            break
    return splits

# 5. Blocked CV with Double Margins
def blocked_cv_double_margin_split(n_samples, n_blocks, gap_before=0, gap_after=0):
    """
    Returns a list of (train_indices, test_indices) for blocked CV with gaps before and after test block.
    """
    block_size = n_samples // n_blocks
    indices = np.arange(n_samples)
    splits = []
    for i in range(n_blocks):
        test_start = i * block_size
        test_end = test_start + block_size
        before = max(0, test_start - gap_before)
        after = min(n_samples, test_end + gap_after)
        train_indices = np.concatenate([
            indices[:before],
            indices[after:]
        ])
        test_indices = indices[test_start:test_end]
        splits.append((train_indices, test_indices))
    return splits

def ridge_regression_fit_torch_nojit(X, y, alpha, batch_size: int = -1):
    if batch_size > 0:
        alpha = alpha.view(-1,1,1).to(X.device, dtype=X.dtype)
    #X = X.float()
    #y = y.float()
    n_features = X.shape[1]
    n_targets = y.shape[1]
    I = torch.eye(n_features, device=X.device, dtype=X.dtype)
    XTX = X.T @ X
    XTy = X.T @ y
    if batch_size == -1:
        w = torch.linalg.solve(XTX + alpha * I, XTy)
        return w  # (n_features, n_targets)
    elif batch_size > 0:
        # Batch processing for large number of targets (25k+ voxels)       
        w = torch.empty((n_features, n_targets), device=X.device, dtype=X.dtype)
        for start in range(0, n_targets, batch_size):
            end = min(start + batch_size, n_targets)
            batch_alpha = alpha[start:end]
            batch_XTy = XTy[:, start:end] # last dim is n_targets
            for i, a in enumerate(batch_alpha):
                w[:, start + i] = torch.linalg.solve(XTX + a * I, batch_XTy[:, i])
        return w


def test_batched_alpha_ridge_regression_fit():
    import torch
    import numpy as np
    from sklearn.linear_model import Ridge

    # Generate synthetic data
    n_samples, n_features, n_targets = 100, 10, 3
    X_np = np.random.randn(n_samples, n_features)
    y_np = np.random.randn(n_samples, n_targets)
    alphas = np.array([0.1, 10.0, 1000.0])

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    alpha = torch.tensor(alphas, dtype=torch.float32)

    # Your function
    w_torch = ridge_regression_fit_torch_nojit(X, y, alpha, batch_size=3).cpu().numpy()

    # scikit-learn reference
    w_sklearn = np.zeros((n_features, n_targets))
    for i in range(n_targets):
        reg = Ridge(alpha=alphas[i], fit_intercept=False)
        reg.fit(X_np, y_np[:, i])
        w_sklearn[:, i] = reg.coef_

    # Compare outputs
    assert np.allclose(w_torch, w_sklearn, atol=1e-6), "Torch and sklearn coefficients do not match!"
    print("Max abs diff batched alphas:", np.abs(w_torch - w_sklearn).max())
#test_batched_alpha_ridge_regression_fit()

#@torch.jit.script
def ridge_regression_fit_torch(X, y, alpha, batch_size: int = 4000):
    alpha = alpha.view(-1,1,1).to(X.device, dtype=X.dtype)
    #X = X.float()
    #y = y.float()
    n_features = X.shape[1]
    n_targets = y.shape[1]
    I = torch.eye(n_features, device=X.device, dtype=X.dtype)
    XTX = X.T @ X
    XTy = X.T @ y
    # Batch processing for large number of targets (25k+ voxels)       
    w = torch.empty((n_features, n_targets), device=X.device, dtype=X.dtype)
    with tqdm(total=n_targets//batch_size, desc="Outer fold regression chunks") as pbar_outer:
        for start in tqdm(range(0, n_targets, batch_size)):
            end = min(start + batch_size, n_targets)
            batch_alpha = alpha[start:end]
            batch_XTy = XTy[:, start:end] # last dim is n_targets
            
            w[:, start:start+batch_size] = torch.linalg.solve(XTX + batch_alpha * I, batch_XTy[:, start:start+batch_size])
            pbar_outer.update()
        return w

import torch
from tqdm import tqdm
@torch.jit.script
def ridge_regression_fit_torch(X, y, alpha, batch_size: int = 8):
    n_features = X.shape[1]
    n_targets = y.shape[1]
    I = torch.eye(n_features, device=X.device, dtype=X.dtype)
    XTX = X.T @ X
    XTy = X.T @ y
    w = torch.empty((n_features, n_targets), device=X.device, dtype=X.dtype)
    for start in range(0, n_targets, batch_size):
        end = min(start + batch_size, n_targets)
        batch_alpha = alpha[start:end].to(X.device, dtype=X.dtype)  # (batch_size,)
        # Build batch of regularized XTX matrices: (batch_size, n_features, n_features)
        batch_XTX = XTX.unsqueeze(0) + batch_alpha.view(-1, 1, 1) * I
        # Build batch of XTy vectors: (batch_size, n_features, 1)
        batch_XTy = XTy[:, start:end].T.unsqueeze(2)
        # Solve in batch
        batch_w = torch.linalg.solve(batch_XTX, batch_XTy).squeeze(2).T  # (n_features, batch_size)
        w[:, start:end] = batch_w
    return w

def ridge_regression_fit_torch(X, y, alpha, batch_size=512):
    """
    X  : (N, F)    features
    y  : (N, T)    many targets
    alpha : (T,)   one λ per target
    chunk : max targets solved together (tune to your GPU RAM)
    returns w : (F, T)
    """
    N, F = X.shape
    T    = y.shape[1]
    XTX  = X.T @ X                                 # (F, F) once
    XTy  = X.T @ y                                 # (F, T) once
    eye  = torch.eye(F, device=X.device, dtype=X.dtype)

    w = torch.empty((F, T), device=X.device, dtype=X.dtype)

    for s in range(0, T, batch_size):
        e        = min(s + batch_size, T)
        lam      = alpha[s:e].to(X.dtype)          # (c,)
        # build (c, F, F) *cheaply* via broadcast; don't keep it afterwards
        K        = XTX[None] + lam[:, None, None] * eye
        rhs      = XTy[:, s:e].T.unsqueeze(2)      # (c, F, 1)
        w[:, s:e] = torch.linalg.solve(K, rhs).squeeze(2).T

    return w

import numpy as np
from sklearn.linear_model import Ridge

def ridge_regression_fit_sklearn(X, y, alpha, batch_size: int = 4000, device='cpu'):
    """
    Ridge regression with separate alpha per target using scikit-learn.

    Parameters:
        X : torch.Tensor or np.ndarray, shape (n_samples, n_features)
            Input feature matrix.
        y : torch.Tensor or np.ndarray, shape (n_samples, n_targets)
            Target matrix.
        alpha : torch.Tensor or np.ndarray, shape (n_targets,)
            Regularization strengths for each target variable.
        batch_size : int, optional (ignored)
            Present for API compatibility; not used.

    Returns:
        w : np.ndarray, shape (n_features, n_targets)
            Weight matrix for each target.
    """
    # Convert to numpy if needed
    device = X.device if isinstance(X, torch.Tensor) else None
    if hasattr(X, 'cpu'):
        X = X.cpu().numpy()
    if hasattr(y, 'cpu'):
        y = y.cpu().numpy()
    if hasattr(alpha, 'cpu'):
        alpha = alpha.cpu().numpy()
    
    model = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    model.fit(X, y)
    # model.coef_ is (n_targets, n_features); transpose to (n_features, n_targets)
    return torch.from_numpy(model.coef_.T).to(device=device, dtype=torch.float32)


def ridge_regression_predict_torch(X, w):
    return X @ w

# --- Test Ridge Regression Implementation ---
def test_ridge_regression_fit():
    np.random.seed(0)
    n_samples, n_features, n_targets = 100, 10, 3
    X_np = np.random.randn(n_samples, n_features)
    y_np = np.random.randn(n_samples, n_targets)
    alpha = np.ones((1, n_targets)) * 0.1  # Regularization strength

    X_torch = torch.tensor(X_np, dtype=torch.float32)
    y_torch = torch.tensor(y_np, dtype=torch.float32)

    w_torch = ridge_regression_fit_torch(X_torch, y_torch, torch.from_numpy(alpha)).cpu().numpy()

    w_sklearn = np.zeros((n_features, n_targets))
    
    for i in range(n_targets):
        w_sklearn[:, i] = sklearn_ridge_regression(X_np, y_np[:, i], alpha=alpha[0,i])

    max_diff = np.max(np.abs(w_torch - w_sklearn))
    print("Max abs difference between PyTorch and sklearn coefficients:", max_diff)
    assert max_diff < 1e-4, "PyTorch and sklearn ridge regression coefficients do not match!"

#test_ridge_regression_fit()


def get_splits(split_function, n_samples, block_labels, n_splits, gap):
    """Helper function to get splits based on the split function type."""
    if split_function == "blocked":
        return blocked_cv_split(n_samples, n_splits, gap=gap)
    elif split_function == "custom_block_split":
        if block_labels is None:
            raise ValueError("block_labels must be provided for custom_block_split.")
        return custom_block_split(block_labels, n_splits, gap=gap)
    else:
        raise NotImplementedError(f"Split function '{split_function}' is not implemented.")

def nested_blocked_cv(X, y, split_function="blocked", block_labels=None, split_indices=None, 
                     n_splits_outer=3, n_splits_inner=2, test_size=200, gap=10, alphas=None, device='cpu'):

    """
    Takes a sequence of model representations and brain representations (time- and lag-aligned) and
    computes an outer cross-validation with N-1 blocks used for training, and 1 block used for testing.
    In the inner cross-validation, the N-1 blocks are again split into train and test. 
    The inner loop creates a score to select the outer loop's ridge regression hyperparameter.
    That parameter is then used in the outer loop to compute the outer test correlation.
    The best outer test-correlation is used.
    """
    n_samples = X.shape[0]
    alphas = alphas if alphas is not None else [0.1, 1.0, 10.0]
    outer_scores = []
    best_alphas = []
    
    # Get outer splits: 4 splits
    outer_splits = get_splits(split_function, n_samples, block_labels, n_splits_outer, gap)
    total_models_to_fit = (n_splits_outer * n_splits_inner + n_splits_outer) * len(alphas)
    with tqdm(total=total_models_to_fit, desc="Ridge Regression models fitted") as pbar:
        for train_idx_outer, test_idx_outer in outer_splits:
            block_labels_outer_train = block_labels[train_idx_outer] if block_labels is not None else None
            block_labels_outer_val = block_labels[test_idx_outer] if block_labels is not None else None

            X_train_outer = X[train_idx_outer]
            y_train_outer = y[train_idx_outer]
            X_test_outer = X[test_idx_outer]
            y_test_outer = y[test_idx_outer]
            
            # Inner CV for hyperparameter selection
            best_alphas_for_fold = [None] * y_train_outer.shape[1]  # one best alpha per voxel
            best_inner_scores = [-float('inf')] * y_train_outer.shape[1]  # one best score per voxel
            
            for alpha in alphas:
                inner_scores = []                
                if split_function == "custom_block_split" and block_labels is not None:
                    # Get the block labels corresponding to the outer training indices
                    original_block_labels = np.array(block_labels)
                    inner_block_labels = original_block_labels[train_idx_outer].tolist()
                else:
                    inner_block_labels = None
                    
                # get inner splits (3) 
                inner_splits = get_splits(split_function, X_train_outer.shape[0], # assume time dimension is first!
                                        inner_block_labels, n_splits_inner, gap)
                # iterate over inner splits
                for train_idx_inner, val_idx_inner in inner_splits:
                    block_labels_inner_train = block_labels_outer_train[train_idx_inner] if block_labels_outer_train is not None else None
                    block_labels_inner_val = block_labels_outer_val[val_idx_inner] if block_labels_outer_val is not None else None
                    X_train_inner = X_train_outer[train_idx_inner]
                    y_train_inner = y_train_outer[train_idx_inner]
                    X_val_inner = X_train_outer[val_idx_inner]
                    y_val_inner = y_train_outer[val_idx_inner]

                    # normalize data per block (experiment run, story, etc.)
                    X_train_inner_normalized, X_means, X_stds = normalize_train_test(X_train_inner, block_ids=block_labels_inner_train)
                    y_train_inner_normalized, y_means, y_stds = normalize_train_test(y_train_inner, block_ids=block_labels_inner_train)
                    # use training data statistics to normalize test data
                    X_val_inner_normalized = (X_val_inner - X_means) / (X_stds + 1e-8)
                    y_val_inner_normalized = (y_val_inner - y_means) / (y_stds + 1e-8)

                    w = ridge_regression_fit_torch_nojit(X_train_inner_normalized, y_train_inner_normalized, alpha, batch_size=-1) # no batching needed here
                    pbar.update()
                    y_pred_inner = ridge_regression_predict_torch(X_val_inner_normalized, w)

                    # Compute mean correlation across voxels for inner split
                    corrs = []
                    corrs, _ = pearson_correlation(y_val_inner_normalized, y_pred_inner)
                    
                    inner_scores.append(corrs.cpu().numpy()) # store inner scores per voxel for this alpha in the inner fold

                # get the best alpha per voxel for the inner fold and update the best_alpha_for_fold.
                print(y_train_outer.shape, "y_train_outer shape")
                for voxel_idx in range(y_train_outer.shape[1]):
                    mean_inner_score = np.mean([score[voxel_idx] for score in inner_scores])
                    if mean_inner_score > best_inner_scores[voxel_idx]:
                        best_inner_scores[voxel_idx] = mean_inner_score
                        best_alphas_for_fold[voxel_idx] = alpha
                print(np.array(best_alphas_for_fold).shape, "best alphas for fold")
            best_alphas.append(best_alphas_for_fold)
            print("Best alphas for fold:", len(best_alphas_for_fold), len([1 for x in best_alphas_for_fold if x is None]), "voxels with no best alpha found")

            print(np.array(best_alphas).shape, "best alphas shape")
            # Train outer model on full outer training set with best alpha found across inner folds
            X_train_outer_normalized, X_means_outer, X_stds_outer = normalize_train_test(X_train_outer, block_ids=block_labels_outer_train)
            y_train_outer_normalized, y_means_outer, y_stds_outer = normalize_train_test(y_train_outer, block_ids=block_labels_outer_train)
            
            # use training data statistics to normalize test data
            y_test_outer_normalized = (y_test_outer - y_means_outer) / (y_stds_outer + 1e-8)
            X_test_outer_normalized = (X_test_outer - X_means_outer) / (X_stds_outer + 1e-8)

            alpha_tensor = torch.from_numpy(np.array(best_alphas_for_fold)).to(device, dtype=torch.float32)
            w_final = ridge_regression_fit_sklearn(X_train_outer_normalized, y_train_outer_normalized, alpha_tensor, batch_size=16)
            y_pred_outer = ridge_regression_predict_torch(X_test_outer_normalized, w_final)
            pbar.update()
            # Evaluate on outer test set
            corrs_outer = []

            corrs_outer, _ = pearson_correlation(y_test_outer_normalized, y_pred_outer)
            
            outer_scores.append(corrs_outer.cpu().numpy())
            print("Outer scores for this fold:", outer_scores[-1])

    best_alphas_for_voxels = []
    best_alphas = np.array(best_alphas)  # Convert to numpy array for easier processing
    for voxel_idx in range(y_train_outer.shape[1]):
        best_alphas_for_voxel = best_alphas[:, voxel_idx].tolist()
        
        if len(best_alphas_for_voxel) > 0:
            alpha_counts = Counter(best_alphas_for_voxel)
            most_common_alpha = alpha_counts.most_common(1)[0][0]
            best_alphas_for_voxels.append(most_common_alpha)
        else:
            print(f"Warning: No best alpha found for voxel {voxel_idx}. Using default alpha {alphas[0]}.")
            best_alphas_for_voxels.append(alphas[0])
    print(outer_scores, "outer scores")
    return best_alphas_for_voxels, outer_scores


