import torch


# =============================================================================
# E-STEP
# =============================================================================

def estimate_log_prob_diag(
    X: torch.Tensor,
    means: torch.Tensor,
    precisions_chol: torch.Tensor,
    degrees_of_freedom: torch.Tensor,
    mean_precision: torch.Tensor,
) -> torch.Tensor:
    """Compute log p(X | Z) for diagonal covariance (E-step).

    Memory-efficient implementation that avoids explicitly materialising
    the large tensor (batch_size, n_components, n_features).

    Parameters
    ----------
    X : Tensor of shape (n, D)
    means : Tensor of shape (K, D)
    precisions_chol : Tensor of shape (K, D)
    degrees_of_freedom : Tensor of shape (K,)
    mean_precision : Tensor of shape (K,)

    Returns
    -------
    log_prob : Tensor of shape (n, K)
    """
    n, D = X.shape
    device = X.device
    dtype = X.dtype

    prec2 = precisions_chol ** 2  # (K, D)

    # log determinant of precision Cholesky: sum_d log(precisions_chol[k, d])
    log_det_chol = torch.sum(torch.log(precisions_chol), dim=1)  # (K,)

    # Mahalanobis without building diff = X[:, None, :] - means[None, :, :]
    # sum_d ((x_d - mu_d) * prec_d)^2
    # = sum_d x_d^2 * prec_d^2 - 2 sum_d x_d * mu_d * prec_d^2 + sum_d mu_d^2 * prec_d^2
    x2_term = (X ** 2) @ prec2.T                                  # (n, K)
    cross_term = X @ (means * prec2).T                            # (n, K)
    mean_term = torch.sum((means ** 2) * prec2, dim=1).unsqueeze(0)  # (1, K)

    maha = x2_term - 2.0 * cross_term + mean_term                 # (n, K)

    const = torch.tensor(2.0 * torch.pi, device=device, dtype=dtype)
    log_prob = -0.5 * (D * torch.log(const) + maha) + log_det_chol.unsqueeze(0)

    # Bayesian correction
    log_prob = log_prob - 0.5 * D * torch.log(degrees_of_freedom).unsqueeze(0)

    # log-lambda
    arange = torch.arange(D, dtype=dtype, device=device)
    log_lambda = D * torch.log(torch.tensor(2.0, device=device, dtype=dtype)) + torch.sum(
        torch.digamma(0.5 * (degrees_of_freedom.unsqueeze(1) - arange.unsqueeze(0))),
        dim=1,
    )  # (K,)

    log_prob = log_prob + 0.5 * (
        log_lambda.unsqueeze(0) - D / mean_precision.unsqueeze(0)
    )

    return log_prob


def estimate_log_prob_full(
    X: torch.Tensor,
    means: torch.Tensor,
    precisions_chol: torch.Tensor,
    degrees_of_freedom: torch.Tensor,
    mean_precision: torch.Tensor,
) -> torch.Tensor:
    """Compute log p(X | Z) for full covariance (E-step)."""
    n, D = X.shape
    device = X.device
    dtype = X.dtype

    log_det_chol = torch.sum(
        torch.log(torch.diagonal(precisions_chol, dim1=1, dim2=2)), dim=1
    )  # (K,)

    diff = X.unsqueeze(1) - means.unsqueeze(0)                    # (n, K, D)
    diff4d = diff.unsqueeze(-1)                                   # (n, K, D, 1)
    prec4d = precisions_chol.unsqueeze(0)                         # (1, K, D, D)
    y = torch.matmul(prec4d.transpose(-1, -2), diff4d)           # (n, K, D, 1)
    maha = (y ** 2).squeeze(-1).sum(dim=2)                       # (n, K)

    const = torch.tensor(2.0 * torch.pi, device=device, dtype=dtype)
    log_prob = -0.5 * (D * torch.log(const) + maha) + log_det_chol.unsqueeze(0)

    log_prob = log_prob - 0.5 * D * torch.log(degrees_of_freedom).unsqueeze(0)

    arange = torch.arange(D, dtype=dtype, device=device)
    log_lambda = D * torch.log(torch.tensor(2.0, device=device, dtype=dtype)) + torch.sum(
        torch.digamma(0.5 * (degrees_of_freedom.unsqueeze(1) - arange.unsqueeze(0))),
        dim=1,
    )

    log_prob = log_prob + 0.5 * (
        log_lambda.unsqueeze(0) - D / mean_precision.unsqueeze(0)
    )

    return log_prob


def estimate_log_prob_spherical(
    X: torch.Tensor,
    means: torch.Tensor,
    precisions_chol: torch.Tensor,
    degrees_of_freedom: torch.Tensor,
    mean_precision: torch.Tensor,
) -> torch.Tensor:
    """Compute log p(X | Z) for spherical covariance (E-step)."""
    n, D = X.shape
    device = X.device
    dtype = X.dtype

    log_det_chol = D * torch.log(precisions_chol)                # (K,)

    diff = X.unsqueeze(1) - means.unsqueeze(0)                   # (n, K, D)
    maha = torch.sum(diff ** 2, dim=2) * (precisions_chol.unsqueeze(0) ** 2)

    const = torch.tensor(2.0 * torch.pi, device=device, dtype=dtype)
    log_prob = -0.5 * (D * torch.log(const) + maha) + log_det_chol.unsqueeze(0)

    log_prob = log_prob - 0.5 * D * torch.log(degrees_of_freedom).unsqueeze(0)

    arange = torch.arange(D, dtype=dtype, device=device)
    log_lambda = D * torch.log(torch.tensor(2.0, device=device, dtype=dtype)) + torch.sum(
        torch.digamma(0.5 * (degrees_of_freedom.unsqueeze(1) - arange.unsqueeze(0))),
        dim=1,
    )

    log_prob = log_prob + 0.5 * (
        log_lambda.unsqueeze(0) - D / mean_precision.unsqueeze(0)
    )

    return log_prob


def estimate_log_weights_dirichlet_process(
    weight_concentration: tuple,
) -> torch.Tensor:
    """Compute log weights for Dirichlet Process prior."""
    alpha, beta = weight_concentration
    digamma_sum = torch.digamma(alpha + beta)
    digamma_a = torch.digamma(alpha)
    digamma_b = torch.digamma(beta)

    cumsum = torch.cat([
        torch.zeros(1, device=alpha.device, dtype=alpha.dtype),
        torch.cumsum(digamma_b - digamma_sum, dim=0)[:-1],
    ])
    return digamma_a - digamma_sum + cumsum


def estimate_log_weights_dirichlet_distribution(
    weight_concentration: torch.Tensor,
) -> torch.Tensor:
    """Compute log weights for Dirichlet Distribution prior."""
    return torch.digamma(weight_concentration) - torch.digamma(
        weight_concentration.sum()
    )


# =============================================================================
# M-STEP
# =============================================================================

def m_step_diag(
    X: torch.Tensor,
    resp: torch.Tensor,
    mean_precision_prior: float,
    mean_prior: torch.Tensor,
    degrees_of_freedom_prior: float,
    covariance_prior: torch.Tensor,
    weight_concentration_prior: float,
    weight_concentration_prior_type: str,
    reg_covar: float,
):
    """Full M-step for diagonal covariance."""
    n, D = X.shape
    K = resp.shape[1]
    eps = torch.finfo(X.dtype).eps

    nk = resp.sum(dim=0).clamp_min(eps)                           # (K,)
    xk = (resp.T @ X) / nk.unsqueeze(1)                           # (K, D)
    x2k = (resp.T @ (X ** 2)) / nk.unsqueeze(1)                   # (K, D)
    sk = x2k - xk ** 2
    sk = sk + reg_covar

    weight_concentration = _update_weight_concentration(
        nk, weight_concentration_prior, weight_concentration_prior_type, K, X.device, X.dtype
    )

    mean_precision = mean_precision_prior + nk
    means = (
        mean_precision_prior * mean_prior + nk.unsqueeze(1) * xk
    ) / mean_precision.unsqueeze(1)

    degrees_of_freedom = degrees_of_freedom_prior + nk

    diff = xk - mean_prior.unsqueeze(0)
    covariances = (
        covariance_prior.unsqueeze(0)
        + nk.unsqueeze(1) * (
            sk + (mean_precision_prior / mean_precision).unsqueeze(1) * diff ** 2
        )
    ) / degrees_of_freedom.unsqueeze(1)

    covariances = covariances.clamp_min(reg_covar)
    precisions_chol = 1.0 / torch.sqrt(covariances)

    return (
        weight_concentration,
        mean_precision,
        means,
        degrees_of_freedom,
        covariances,
        precisions_chol,
    )


def m_step_full(
    X: torch.Tensor,
    resp: torch.Tensor,
    mean_precision_prior: float,
    mean_prior: torch.Tensor,
    degrees_of_freedom_prior: float,
    covariance_prior: torch.Tensor,
    weight_concentration_prior: float,
    weight_concentration_prior_type: str,
    reg_covar: float,
):
    """Full M-step for full covariance."""
    n, D = X.shape
    K = resp.shape[1]
    eps = torch.finfo(X.dtype).eps

    nk = resp.sum(dim=0).clamp_min(eps)
    xk = (resp.T @ X) / nk.unsqueeze(1)

    diff_x = X.unsqueeze(1) - xk.unsqueeze(0)                     # (n, K, D)
    w_diff = resp.unsqueeze(2) * diff_x
    sk = torch.einsum('nki,nkj->kij', w_diff, diff_x) / nk.reshape(K, 1, 1)
    sk = sk + reg_covar * torch.eye(D, device=X.device, dtype=X.dtype).unsqueeze(0)

    weight_concentration = _update_weight_concentration(
        nk, weight_concentration_prior, weight_concentration_prior_type, K, X.device, X.dtype
    )

    mean_precision = mean_precision_prior + nk
    means = (
        mean_precision_prior * mean_prior + nk.unsqueeze(1) * xk
    ) / mean_precision.unsqueeze(1)

    degrees_of_freedom = degrees_of_freedom_prior + nk

    diff_mu = xk - mean_prior.unsqueeze(0)
    outer = torch.einsum('ki,kj->kij', diff_mu, diff_mu)

    covariances = (
        covariance_prior.unsqueeze(0)
        + nk.reshape(K, 1, 1) * sk
        + (nk * mean_precision_prior / mean_precision).reshape(K, 1, 1) * outer
    ) / degrees_of_freedom.reshape(K, 1, 1)

    covariances = covariances + reg_covar * torch.eye(
        D, device=X.device, dtype=X.dtype
    ).unsqueeze(0)

    precisions_chol = _batch_cholesky_precision(covariances)

    return (
        weight_concentration,
        mean_precision,
        means,
        degrees_of_freedom,
        covariances,
        precisions_chol,
    )


def m_step_spherical(
    X: torch.Tensor,
    resp: torch.Tensor,
    mean_precision_prior: float,
    mean_prior: torch.Tensor,
    degrees_of_freedom_prior: float,
    covariance_prior: float,
    weight_concentration_prior: float,
    weight_concentration_prior_type: str,
    reg_covar: float,
):
    """Full M-step for spherical covariance."""
    n, D = X.shape
    K = resp.shape[1]
    eps = torch.finfo(X.dtype).eps

    nk = resp.sum(dim=0).clamp_min(eps)
    xk = (resp.T @ X) / nk.unsqueeze(1)
    x2k = (resp.T @ (X ** 2)) / nk.unsqueeze(1)
    sk = (x2k - xk ** 2).mean(dim=1) + reg_covar

    weight_concentration = _update_weight_concentration(
        nk, weight_concentration_prior, weight_concentration_prior_type, K, X.device, X.dtype
    )

    mean_precision = mean_precision_prior + nk
    means = (
        mean_precision_prior * mean_prior + nk.unsqueeze(1) * xk
    ) / mean_precision.unsqueeze(1)

    degrees_of_freedom = degrees_of_freedom_prior + nk

    diff = xk - mean_prior.unsqueeze(0)
    covariances = (
        covariance_prior
        + nk * (sk + mean_precision_prior / mean_precision * diff.pow(2).mean(dim=1))
    ) / degrees_of_freedom

    covariances = covariances.clamp_min(reg_covar)
    precisions_chol = 1.0 / torch.sqrt(covariances)

    return (
        weight_concentration,
        mean_precision,
        means,
        degrees_of_freedom,
        covariances,
        precisions_chol,
    )


# =============================================================================
# HELPERS
# =============================================================================

def _update_weight_concentration(nk, prior, prior_type, K, device, dtype):
    """Compute updated weight concentration for either prior type."""
    if prior_type == "dirichlet_process":
        alpha = 1.0 + nk
        cumrev = torch.cumsum(nk.flip(0), dim=0).flip(0)
        beta = prior + torch.cat([
            cumrev[1:],
            torch.zeros(1, device=device, dtype=dtype)
        ])
        return (alpha, beta)
    return prior + nk


def _batch_cholesky_precision(covariances: torch.Tensor) -> torch.Tensor:
    """Compute lower-triangular Cholesky of precision = inv(covariance)."""
    cov_chol = torch.linalg.cholesky(covariances)
    K, D, _ = cov_chol.shape
    eye = torch.eye(
        D,
        device=covariances.device,
        dtype=covariances.dtype
    ).unsqueeze(0).expand(K, -1, -1)

    L_inv = torch.linalg.solve_triangular(cov_chol, eye, upper=False)
    return L_inv.transpose(-1, -2)


def to_tensor(arr, device: torch.device, dtype: torch.dtype = torch.float32):
    """Convert a numpy array or tensor to a tensor on device."""
    import numpy as np

    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(arr), device=device, dtype=dtype)