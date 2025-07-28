import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

import numpy as np
import h5py
import os
import torch
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Union
import numexpr as ne

import torch
from torch import Tensor
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
import torch
from torch import Tensor
from scipy import stats
from scipy.stats import pearsonr          # SciPy ≥ 1.7 (vectorised)

# ---------- helper: Student-t CDF on the same device ------------------------

def _student_t_cdf(t: Tensor, df: int) -> Tensor:

    return torch.as_tensor(stats.t.cdf(t.cpu().numpy(), df), device=t.device, dtype=torch.float32)


# ---------- column-wise Pearson r *and* p-value ------------------------------
def pearson_correlation(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """
    Returns (r, p) for each column of two (T, N) tensors.
    * No ε bias     * Works on CPU or GPU     * Matches SciPy exactly
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shape (T, N)")

    # centre once
    xm = x - x.mean(dim=0, keepdim=True)
    ym = y - y.mean(dim=0, keepdim=True)

    # Pearson r
    r_num = (xm * ym).sum(dim=0)                              # covariance * (T-1)
    r_den = torch.sqrt((xm**2).sum(dim=0) * (ym**2).sum(dim=0))
    r = r_num / r_den                                         # shape (N,)

    # p-value via Student-t test
    df = x.shape[0] - 2
    t_stat = r * torch.sqrt(df / (1.0 - r**2))
    p = 2.0 * (1.0 - _student_t_cdf(torch.abs(t_stat), df))   # two-sided

    return r, p

import torch
import numpy as np
from scipy import stats

def test_pearson_correlation():
    # Set reproducibility
    np.random.seed(42)
    T = 100  # samples
    N = 3    # columns

    # Generate x: each column is a feature
    x = np.random.normal(0, 1, (T, N))

    # y0: strong positive correlation with x0
    y0 = x[:, 0] * 0.8 + np.random.normal(0, 0.2, T)
    # y1: strong negative correlation with x1
    y1 = x[:, 1] * -0.5 + np.random.normal(0, 0.3, T)
    # y2: no correlation with x2
    y2 = np.random.normal(0, 1, T)

    y = np.stack([y0, y1, y2], axis=1)

    # Convert to torch tensors
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # Call your function
    r, p = pearson_correlation(x_t, y_t)

    # Compare to scipy.stats.pearsonr for each column
    for i in range(N):
        r_scipy, p_scipy = stats.pearsonr(x[:, i], y[:, i])
        assert np.isclose(r[i].item(), r_scipy, atol=1e-5), f"Correlation mismatch at col {i}"
        assert np.isclose(p[i].item(), p_scipy, atol=1e-5), f"P-value mismatch at col {i}"

    print("All tests passed for pearson_correlation function.")

test_pearson_correlation()

""" Test passes: pearson correlation works - both PyTorch and SciPy match to single-precision accuracy and p-values are computed correctly.
import torch
from torch.testing import assert_close
from scipy.stats import pearsonr

# ------------------------------------------------------------------
# 1.  Random data – single-precision
# ------------------------------------------------------------------
T, N = 1000, 8
torch.manual_seed(0)
x = torch.randn(T, N, dtype=torch.float32)
y = torch.randn(T, N, dtype=torch.float32)

# ------------------------------------------------------------------
# 2.  Your PyTorch implementation  (returns float32 tensors)
# ------------------------------------------------------------------
r_torch, p_torch = pearson_correlation(x, y)    # <-- your function

# ------------------------------------------------------------------
# 3.  SciPy reference   (cast to the *same* dtype)
# ------------------------------------------------------------------
r_np, p_np = pearsonr(x.numpy(), y.numpy(), axis=0)
r_ref = torch.from_numpy(r_np.astype('float32'))
p_ref = torch.from_numpy(p_np.astype('float32'))

# ------------------------------------------------------------------
# 4.  Assertions with dtype-appropriate tolerances
# ------------------------------------------------------------------
assert_close(r_torch, r_ref, rtol=1e-5, atol=1e-7)
assert_close(p_torch, p_ref, rtol=1e-5, atol=1e-7)

print("✅  PyTorch and SciPy match to single-precision accuracy.")
"""


def permutation_test_with_correction(predictions, # n_voxels, n_TRs
                                     ground_truth, # n_voxels, n_TRs
                                     k_steps=1000, # number of resamples
                                     block_size=20, # permute in blocks (1 = TR-wise permutation)
                                     alpha=0.05, # correction parameter
                                     correction_type="fdr_bh" # correction type
                                    ): 
    """    
    predictions: prediction matrix from the model (n_voxels, n_TRs)
    ground_truth: ground-truth fMRI data matrix (n_voxels, n_TRs)
    k_steps: number of permuted resamples to construct the null-distribution (1000+)
    block_size: permutation is done in blocks, not at TR level. Usually 10 or 20 TRs.
    alpha: multiple corrections threshold (e.g. 0.05)
    correction_type: method for multiple corrections. "holm" or "fdr_bh" are commonly used.

    returns:
    corrected_p_values (per voxel)
    """
    n_voxels, n_timesteps = predictions.shape

    # Step 3: Compute original correlations for each voxel
    original_corrs = np.array([
        pearsonr(predictions[v], ground_truth[v])[0]
        for v in range(n_voxels)
    ])

    # Prepare to store permuted correlations
    permuted_corrs = np.zeros((k_steps, n_voxels))

    # Step 4: Loop over k-steps
    for i in range(k_steps):
        permuted_predictions = np.copy(predictions)
        n_blocks = n_timesteps // block_size
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size
            # Permute the block along the time axis (block-wise permutation)
            permuted_block = np.random.permutation(permuted_predictions[:, start:end].T).T
            permuted_predictions[:, start:end] = permuted_block

        # Step 4.1: Compute correlations for permuted predictions
        # permuted_corrs[i] = pearson_correlation(permuted_predictions, ground_truth)[0].cpu().numpy().tolist()  # This computes the correlation for each voxel
        permuted_corrs[i] = pearson_correlation(
            torch.tensor(permuted_predictions), torch.tensor(ground_truth)
        )[0].cpu().numpy().tolist()
        #for v in range(n_voxels):
        #    permuted_corrs[i, v] = pearsonr(permuted_predictions[v], ground_truth[v])[0]

    # Step 5: Compute p-values for each voxel
    p_values = np.zeros(n_voxels)
    for v in range(n_voxels):
        null_dist = permuted_corrs[:, v]
        # p-value: fraction of permuted correlations >= original correlation (one-sided)
        p_values[v] = (np.sum(null_dist >= original_corrs[v]) + 1) / (k_steps + 1)  # +1 for continuity correction

    # multipletests returns: reject, pvals_corrected, alphacSidak, alphacBonf
    _, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=correction_type)  # 'holm' for Holm-Bonferroni

    return pvals_corrected

