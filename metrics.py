import torch

def compute_ranks_from_features(feature_matrices):
    """Computes different approximations of the rank of the feature matrices.

    Args:
        feature_matrices (torch.Tensor): A tensor of shape (B_matrices, N_obs, D_dims).

    (1) Effective rank.
    A continuous approximation of the rank of a matrix.
    Definition 2.1. in Roy & Vetterli, (2007) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7098875
    Also used in Huh et al. (2023) https://arxiv.org/pdf/2103.10427.pdf


    (2) Approximate rank.
    Threshold at the dimensions explaining 99% of the variance in a PCA analysis.
    Section 2 in Yang et al. (2020) https://arxiv.org/pdf/1909.12255.pdf

    (3) srank.
    Another (incorrect?) version of (2).
    Section 3 in Kumar et al. https://arxiv.org/pdf/2010.14498.pdf

    (4) Feature rank.
    A threshold rank: normalize by dim size and discard dimensions with singular values below 0.01.
    Equations (4) and (5). Lyle et al. (2022) https://arxiv.org/pdf/2204.09560.pdf

    (5) PyTorch/NumPy rank.
    Rank defined in PyTorch and NumPy (https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html)
    (https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html)
    Quoting Numpy:
        This is the algorithm MATLAB uses [1].
        It also appears in Numerical recipes in the discussion of SVD solutions for linear least squares [2].
        [1] MATLAB reference documentation, “Rank” https://www.mathworks.com/help/techdoc/ref/rank.html
        [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, “Numerical Recipes (3rd edition)”,
        Cambridge University Press, 2007, page 795.

    """
    cutoff = 0.01  # not used in (1), 1 - 99% in (2), delta in (3), epsilon in (4).
    threshold = 1 - cutoff

    if feature_matrices.shape[1] < feature_matrices.shape[2]:
        return {}  # N < D.

    svals = torch.linalg.svdvals(feature_matrices)

    # (1) Effective rank. Roy & Vetterli (2007)
    sval_sum = torch.sum(svals, dim=1)
    sval_dist = svals / sval_sum.unsqueeze(-1)
    # Replace 0 with 1. This is a safe trick to avoid log(0) = -inf
    # as Roy & Vetterli assume 0*log(0) = 0 = 1*log(1).
    sval_dist_fixed = torch.where(sval_dist == 0, torch.ones_like(sval_dist), sval_dist)
    effective_ranks = torch.exp(-torch.sum(sval_dist_fixed * torch.log(sval_dist_fixed), dim=1))

    # (2) Approximate rank. PCA variance. Yang et al. (2020)
    sval_squares = svals**2
    sval_squares_sum = torch.sum(sval_squares, dim=1)
    cumsum_squares = torch.cumsum(sval_squares, dim=1)
    threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum.unsqueeze(-1))
    approximate_ranks = (~threshold_crossed).sum(dim=-1) + 1

    # (3) srank. Weird. Kumar et al. (2020)
    cumsum = torch.cumsum(svals, dim=1)
    threshold_crossed = cumsum >= threshold * sval_sum.unsqueeze(-1)
    sranks = (~threshold_crossed).sum(dim=-1) + 1

    # (4) Feature rank. Most basic. Lyle et al. (2022)
    n_obs = torch.tensor(feature_matrices.shape[1], device=feature_matrices.device)
    svals_of_normalized = svals / torch.sqrt(n_obs)
    over_cutoff = svals_of_normalized > cutoff
    feature_ranks = over_cutoff.sum(dim=-1)

    # (5) PyTorch/NumPy rank.
    pytorch_ranks = torch.linalg.matrix_rank(feature_matrices)

    # Some singular values.
    singular_values = dict(
        lambda_1=svals_of_normalized[:, 0],
        lambda_N=svals_of_normalized[:, -1],
    )
    if svals_of_normalized.shape[1] > 1:
        singular_values.update(lambda_2=svals_of_normalized[:, 1])

    ranks = dict(
        effective_rank_vetterli=effective_ranks,
        approximate_rank_pca=approximate_ranks,
        srank_kumar=sranks,
        feature_rank_lyle=feature_ranks,
        pytorch_rank=pytorch_ranks,
    )

    out = {**singular_values, **ranks}

    return out

def compute_effective_ranks(data_list, data_groups, data_features):
    """
    Computes the effective ranks of each of the data_features[i][j] in data_list[i] in a batched way.
    Expects flat (no time dimension) tensordicts containing a feature matrix per data_features[i][j].

    The method Does not support different feature matrix shapes.
    """
    stack = [data[data_feature] for i, data in enumerate(data_list) for data_feature in data_features[i]]
    groups = [f"{data_feature}_{group}" for i, group in enumerate(data_groups) for data_feature in data_features[i]]
    features = torch.stack(stack, dim=0)
    try:
        ranks = compute_ranks_from_features(features)
    except torch._C._LinAlgError:
        return {}
    out = {}
    for rank_group, ranks_values in ranks.items():
        for i, data_feature_group in enumerate(groups):
            out[f"SVD/{rank_group}/{data_feature_group}"] = ranks_values[i]
    return out

def compute_dead_neurons_from_features(features, activation):
    TANH_STD_THRESHOLD = 0.001
    match activation:
        case "ReLU" | "GELU":
            return (features == 0).all(dim=1).sum(dim=-1)
        case "Tanh":
            return (features.std(dim=1) < TANH_STD_THRESHOLD).sum(dim=-1)
        case "LeakyReLU":
            return (features < 0).all(dim=1).sum(dim=-1)
        case other:
            raise NotImplementedError(f"Activation {activation} not implemented.")