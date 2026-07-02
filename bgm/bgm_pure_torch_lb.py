"""
bgm_pure_torch.py
-----------------
Pure PyTorch implementation of Bayesian Gaussian Mixture Model.

The entire variational EM loop runs on GPU — no numpy overhead between iterations.
Sklearn is used only for k-means++ initialization, then everything
stays on the GPU until convergence.

Public interface mirrors sklearn's BayesianGaussianMixture:
    fit(X)
    predict(X)
    predict_proba(X)
    fit_predict(X)
    weights_
    means_
    covariances_
    converged_
    n_iter_
    lower_bound_

Supported covariance types: 'diag', 'full', 'spherical'.
Supported weight priors: 'dirichlet_process', 'dirichlet_distribution'.
"""

import warnings
import numpy as np
import torch

from .bgm_torch_ops import (
    estimate_log_prob_diag,
    estimate_log_prob_full,
    estimate_log_prob_spherical,
    estimate_log_weights_dirichlet_process,
    estimate_log_weights_dirichlet_distribution,
    m_step_diag,
    m_step_full,
    m_step_spherical,
    to_tensor,
)
from .memory_utils import resolve_batch_size, iter_batches


class BayesianGaussianMixtureTorch:
    """Pure PyTorch Bayesian Gaussian Mixture Model."""

    def __init__(
        self,
        *,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        means_init=None,                          # NEW: custom means initialization
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None,
        mean_prior=None,
        degrees_of_freedom_prior=None,
        covariance_prior=None,
        random_state=None,
        verbose=0,
        verbose_interval=10,
        batch_size=None,
        device="cuda",
        dtype=torch.float32,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.means_init = means_init              # NEW
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.batch_size = batch_size
        self.dtype = dtype

        if isinstance(device, str):
            if device == "cuda" and not torch.cuda.is_available():
                warnings.warn("CUDA not available, falling back to CPU.")
                device = "cpu"
            self.device = torch.device(device)
        else:
            self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """Fit the model to X."""
        X_np = np.asarray(X, dtype=np.float32)
        self._check_parameters(X_np)

        rng = np.random.default_rng(self.random_state)

        best_lower_bound = -np.inf
        best_params = None
        best_n_iter = 0
        best_converged = False

        X_gpu = to_tensor(X_np, device=self.device, dtype=self.dtype)

        for init in range(self.n_init):
            if self.verbose >= 1:
                print(f"Initialization {init + 1}/{self.n_init}")

            params = self._initialize(X_np, X_gpu, rng)
            lower_bound = -np.inf
            converged = False

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_resp, log_prob_norm = self._e_step(X_gpu, params)
                params = self._m_step_gpu(X_gpu, log_resp)

                lower_bound = self._compute_lower_bound_gpu(
                    log_resp=log_resp,
                    log_prob_norm=log_prob_norm,
                    params=params,
                )

                if not np.isfinite(lower_bound):
                    raise FloatingPointError(
                        "Lower bound became non-finite. "
                        "Check responsibilities / precisions / priors."
                    )

                change = lower_bound - prev_lower_bound

                if self.verbose >= 2 and n_iter % self.verbose_interval == 0:
                    print(f"  Iteration {n_iter}\t lower bound change {change:.6f}")
                elif self.verbose == 1 and n_iter % self.verbose_interval == 0:
                    print(f"  Iteration {n_iter}")

                if np.isfinite(change) and abs(change) < self.tol:
                    converged = True
                    break

            if lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                best_params = params
                best_n_iter = n_iter
                best_converged = converged

        if best_params is None:
            raise RuntimeError("Model failed to produce valid parameters.")

        if not best_converged:
            warnings.warn(
                "Best performing initialization did not converge. "
                "Try different init parameters, increase max_iter, or adjust tol.",
                stacklevel=2,
            )

        self.converged_ = best_converged
        self._set_attributes(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = float(best_lower_bound)

        return self

    def predict(self, X):
        """Predict cluster labels for X."""
        X_gpu = to_tensor(np.asarray(X, dtype=np.float32), self.device, self.dtype)
        log_resp, _ = self._e_step(X_gpu, self._params_from_attributes())
        return log_resp.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        """Predict posterior probabilities for X."""
        X_gpu = to_tensor(np.asarray(X, dtype=np.float32), self.device, self.dtype)
        log_resp, _ = self._e_step(X_gpu, self._params_from_attributes())
        return log_resp.exp().cpu().numpy()

    def fit_predict(self, X, y=None):
        """Fit and return cluster labels."""
        self.fit(X, y)
        return self.predict(X)

    # ------------------------------------------------------------------
    # Parameter checks and initialization
    # ------------------------------------------------------------------

    def _check_parameters(self, X):
        _, n_features = X.shape

        if self.covariance_type not in {"diag", "full", "spherical"}:
            raise ValueError(
                "covariance_type must be one of {'diag', 'full', 'spherical'}"
            )

        if self.weight_concentration_prior_type not in {
            "dirichlet_process",
            "dirichlet_distribution",
        }:
            raise ValueError(
                "weight_concentration_prior_type must be "
                "'dirichlet_process' or 'dirichlet_distribution'"
            )

        if self.weight_concentration_prior is None:
            self.weight_concentration_prior_ = 1.0 / self.n_components
        else:
            self.weight_concentration_prior_ = float(self.weight_concentration_prior)

        self.mean_precision_prior_ = float(
            1.0 if self.mean_precision_prior is None else self.mean_precision_prior
        )

        if self.mean_prior is None:
            self.mean_prior_ = X.mean(axis=0).astype(np.float32)
        else:
            self.mean_prior_ = np.asarray(self.mean_prior, dtype=np.float32)

        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = float(n_features)
        else:
            if self.degrees_of_freedom_prior <= n_features - 1:
                raise ValueError(
                    f"degrees_of_freedom_prior must be > {n_features - 1}"
                )
            self.degrees_of_freedom_prior_ = float(self.degrees_of_freedom_prior)

        if self.covariance_prior is None:
            if self.covariance_type == "full":
                self.covariance_prior_ = np.cov(X.T).astype(np.float32)
            elif self.covariance_type == "diag":
                self.covariance_prior_ = np.var(X, axis=0, ddof=1).astype(np.float32)
            else:
                self.covariance_prior_ = float(np.var(X, axis=0, ddof=1).mean())
        else:
            if self.covariance_type == "spherical":
                self.covariance_prior_ = float(self.covariance_prior)
            else:
                self.covariance_prior_ = np.asarray(
                    self.covariance_prior, dtype=np.float32
                )

    def _initialize(self, X_np, X_gpu, rng):
        """Initialize parameters using k-means++, random, or custom responsibilities."""
        n_samples, _ = X_np.shape
        K = self.n_components

        if self.init_params in ("kmeans", "k-means++"):
            from sklearn.cluster import KMeans

            km = KMeans(
                n_clusters=K,
                n_init=1,
                init="k-means++",
                random_state=self.random_state,
            ).fit(X_np)
            labels = km.labels_
            resp_np = np.zeros((n_samples, K), dtype=np.float32)
            resp_np[np.arange(n_samples), labels] = 1.0

        elif self.init_params == "random":
            resp_np = rng.uniform(size=(n_samples, K)).astype(np.float32)
            resp_np /= resp_np.sum(axis=1, keepdims=True)

        elif self.init_params == "custom":
            # NEW: use provided means_init to assign each sample to nearest centroid
            if self.means_init is None:
                raise ValueError(
                    "means_init must be provided when init_params='custom'."
                )
            means_init_np = np.asarray(self.means_init, dtype=np.float32)
            if means_init_np.shape[0] != K:
                raise ValueError(
                    f"means_init has {means_init_np.shape[0]} rows but "
                    f"n_components={K}. Shapes must match."
                )
            from sklearn.metrics import pairwise_distances_argmin
            labels = pairwise_distances_argmin(X_np, means_init_np)
            resp_np = np.zeros((n_samples, K), dtype=np.float32)
            resp_np[np.arange(n_samples), labels] = 1.0

        else:
            raise ValueError(
                "init_params must be 'kmeans', 'k-means++', 'random', or 'custom'."
            )

        resp_gpu = to_tensor(resp_np, device=self.device, dtype=self.dtype)
        return self._m_step_gpu(X_gpu, torch.log(resp_gpu + 1e-10))

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _e_step(self, X_gpu, params):
        """Compute log responsibilities and log_prob_norm on GPU."""
        n_samples = X_gpu.shape[0]
        K = self.n_components

        (
            weight_concentration,
            mean_precision,
            means,
            degrees_of_freedom,
            _covariances,
            precisions_chol,
        ) = params

        if self.weight_concentration_prior_type == "dirichlet_process":
            log_weights = estimate_log_weights_dirichlet_process(weight_concentration)
        else:
            log_weights = estimate_log_weights_dirichlet_distribution(
                weight_concentration
            )

        _log_prob_fn = {
            "diag": estimate_log_prob_diag,
            "full": estimate_log_prob_full,
            "spherical": estimate_log_prob_spherical,
        }[self.covariance_type]

        batch_size = resolve_batch_size(
            batch_size=self.batch_size,
            n_samples=n_samples,
            n_components=K,
            n_features=means.shape[1],
            device=self.device,
            covariance_type=self.covariance_type,
            dtype=self.dtype,
        )

        if batch_size >= n_samples:
            log_prob = _log_prob_fn(
                X_gpu, means, precisions_chol, degrees_of_freedom, mean_precision
            )
            weighted = log_prob + log_weights.unsqueeze(0)
            log_prob_norm = torch.logsumexp(weighted, dim=1)
            log_resp = weighted - log_prob_norm.unsqueeze(1)
        else:
            log_resp_cpu = torch.empty((n_samples, K), dtype=self.dtype, device="cpu")
            log_prob_norm_cpu = torch.empty(n_samples, dtype=self.dtype, device="cpu")

            for X_batch, start, end in iter_batches(X_gpu, batch_size):
                lp = _log_prob_fn(
                    X_batch,
                    means,
                    precisions_chol,
                    degrees_of_freedom,
                    mean_precision,
                )
                weighted = lp + log_weights.unsqueeze(0)
                lpn = torch.logsumexp(weighted, dim=1)
                log_resp_cpu[start:end] = (weighted - lpn.unsqueeze(1)).cpu()
                log_prob_norm_cpu[start:end] = lpn.cpu()

            log_resp = log_resp_cpu.to(self.device)
            log_prob_norm = log_prob_norm_cpu.to(self.device)

        return log_resp, log_prob_norm

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def _m_step_gpu(self, X_gpu, log_resp):
        """Update all parameters on GPU."""
        resp = torch.exp(log_resp)
        mean_prior = to_tensor(self.mean_prior_, self.device, self.dtype)

        if self.covariance_type == "diag":
            cov_prior = to_tensor(self.covariance_prior_, self.device, self.dtype)
            return m_step_diag(
                X=X_gpu,
                resp=resp,
                mean_precision_prior=self.mean_precision_prior_,
                mean_prior=mean_prior,
                degrees_of_freedom_prior=self.degrees_of_freedom_prior_,
                covariance_prior=cov_prior,
                weight_concentration_prior=self.weight_concentration_prior_,
                weight_concentration_prior_type=self.weight_concentration_prior_type,
                reg_covar=self.reg_covar,
            )

        if self.covariance_type == "full":
            cov_prior = to_tensor(self.covariance_prior_, self.device, self.dtype)
            return m_step_full(
                X=X_gpu,
                resp=resp,
                mean_precision_prior=self.mean_precision_prior_,
                mean_prior=mean_prior,
                degrees_of_freedom_prior=self.degrees_of_freedom_prior_,
                covariance_prior=cov_prior,
                weight_concentration_prior=self.weight_concentration_prior_,
                weight_concentration_prior_type=self.weight_concentration_prior_type,
                reg_covar=self.reg_covar,
            )

        return m_step_spherical(
            X=X_gpu,
            resp=resp,
            mean_precision_prior=self.mean_precision_prior_,
            mean_prior=mean_prior,
            degrees_of_freedom_prior=self.degrees_of_freedom_prior_,
            covariance_prior=float(self.covariance_prior_),
            weight_concentration_prior=self.weight_concentration_prior_,
            weight_concentration_prior_type=self.weight_concentration_prior_type,
            reg_covar=self.reg_covar,
        )

    # ------------------------------------------------------------------
    # Lower bound
    # ------------------------------------------------------------------

    def _compute_lower_bound_gpu(self, log_resp, log_prob_norm, params):
        """Compute lower bound aligned with sklearn BayesianGaussianMixture."""
        (
            weight_concentration,
            mean_precision,
            _means,
            degrees_of_freedom,
            _covariances,
            precisions_chol,
        ) = params

        n_features = self.mean_prior_.shape[0]
        device = self.device
        dtype = self.dtype

        if self.covariance_type == "diag":
            log_det_precisions_chol = torch.sum(torch.log(precisions_chol), dim=1)
        elif self.covariance_type == "full":
            log_det_precisions_chol = torch.sum(
                torch.log(torch.diagonal(precisions_chol, dim1=1, dim2=2)),
                dim=1,
            )
        elif self.covariance_type == "spherical":
            log_det_precisions_chol = n_features * torch.log(precisions_chol)
        else:
            raise ValueError(
                f"Unsupported covariance_type: {self.covariance_type!r}"
            )

        log_det_precisions_chol = (
            log_det_precisions_chol
            - 0.5 * n_features * torch.log(degrees_of_freedom)
        )

        arange = torch.arange(n_features, device=device, dtype=dtype).unsqueeze(1)

        log_wishart_per_component = -(
            degrees_of_freedom * log_det_precisions_chol
            + degrees_of_freedom
            * n_features
            * 0.5
            * torch.log(torch.tensor(2.0, device=device, dtype=dtype))
            + torch.sum(
                torch.lgamma(0.5 * (degrees_of_freedom.unsqueeze(0) - arange)),
                dim=0,
            )
        )
        log_wishart = torch.sum(log_wishart_per_component)

        if self.weight_concentration_prior_type == "dirichlet_process":
            alpha, beta = weight_concentration
            log_norm_weight = -torch.sum(
                torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
            )
        else:
            wc = weight_concentration
            log_norm_weight = torch.lgamma(torch.sum(wc)) - torch.sum(torch.lgamma(wc))

        entropy_term = -torch.sum(torch.exp(log_resp) * log_resp)

        lower_bound = (
            entropy_term
            - log_wishart
            - log_norm_weight
            - 0.5 * n_features * torch.sum(torch.log(mean_precision))
        )

        return float(lower_bound.item())

    # ------------------------------------------------------------------
    # Attribute management
    # ------------------------------------------------------------------

    def _set_attributes(self, params):
        """Convert GPU tensors to numpy and set sklearn-compatible attributes."""
        (
            weight_concentration,
            mean_precision,
            means,
            degrees_of_freedom,
            covariances,
            precisions_chol,
        ) = params

        def _np(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy().astype(np.float64)
            return t

        self.mean_precision_ = _np(mean_precision)
        self.means_ = _np(means)
        self.degrees_of_freedom_ = _np(degrees_of_freedom)
        self.covariances_ = _np(covariances)
        self.precisions_cholesky_ = _np(precisions_chol)

        if self.weight_concentration_prior_type == "dirichlet_process":
            alpha, beta = weight_concentration
            self.weight_concentration_ = (_np(alpha), _np(beta))
            weight_dirichlet_sum = (
                self.weight_concentration_[0] + self.weight_concentration_[1]
            )
            tmp = self.weight_concentration_[1] / weight_dirichlet_sum
            self.weights_ = (
                self.weight_concentration_[0]
                / weight_dirichlet_sum
                * np.hstack((1.0, np.cumprod(tmp[:-1])))
            )
            self.weights_ /= self.weights_.sum()
        else:
            self.weight_concentration_ = _np(weight_concentration)
            self.weights_ = self.weight_concentration_ / self.weight_concentration_.sum()

    def _params_from_attributes(self):
        """Reconstruct GPU parameter tuple from numpy attributes."""
        def _t(arr):
            return to_tensor(np.asarray(arr, dtype=np.float32), self.device, self.dtype)

        if self.weight_concentration_prior_type == "dirichlet_process":
            wc = (_t(self.weight_concentration_[0]), _t(self.weight_concentration_[1]))
        else:
            wc = _t(self.weight_concentration_)

        return (
            wc,
            _t(self.mean_precision_),
            _t(self.means_),
            _t(self.degrees_of_freedom_),
            _t(self.covariances_),
            _t(self.precisions_cholesky_),
        )


def _match_components_by_mean(cpu_means, gpu_means):
    """Match components by Euclidean distance between means."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        n = min(cpu_means.shape[0], gpu_means.shape[0])
        return np.arange(n), np.arange(n)

    diff = cpu_means[:, None, :] - gpu_means[None, :, :]
    cost = np.sqrt((diff ** 2).sum(axis=2))
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind


# =============================================================================
# Quick sanity check
# =============================================================================
if __name__ == "__main__":
    import time
    from sklearn.mixture import BayesianGaussianMixture

    rng = np.random.default_rng(42)

    configs = [
        (100_000, 50, 20, "small  (100k, 50f, 20k)"),
        (500_000, 50, 50, "medium (500k, 50f, 50k)"),
        (1_000_000, 10, 45, "real   (1M, 10f, 45k)"),
    ]

    for n, D, K, label in configs:
        print(f"\n{'=' * 60}")
        print(f"Config: {label}")
        print(f"{'=' * 60}")
        X = rng.standard_normal((n, D)).astype(np.float32)

        print("--- sklearn (CPU) ---")
        skl = BayesianGaussianMixture(
            n_components=K,
            covariance_type="diag",
            max_iter=10,
            random_state=42,
        )
        t0 = time.perf_counter()
        skl.fit(X)
        t_skl = time.perf_counter() - t0
        print(f"  time: {t_skl:.2f}s")

        print("--- Pure Torch (GPU) ---")
        gpu = BayesianGaussianMixtureTorch(
            n_components=K,
            covariance_type="diag",
            max_iter=10,
            random_state=42,
            batch_size=None,
            device="cuda",
        )
        t0 = time.perf_counter()
        gpu.fit(X)
        t_gpu = time.perf_counter() - t0
        print(f"  time: {t_gpu:.2f}s")
        print(f"  Speedup: {t_skl / t_gpu:.2f}x")

        row_ind, col_ind = _match_components_by_mean(skl.means_, gpu.means_)
        diff = np.abs(skl.means_[row_ind] - gpu.means_[col_ind]).max()
        mean_diff = np.abs(skl.means_[row_ind] - gpu.means_[col_ind]).mean()

        print(f"  Mean abs diff means: {mean_diff:.6f}")
        print(f"  Max abs diff means:  {diff:.6f}")
        print("  Numerical check:", "OK" if mean_diff < 0.1 else "WARNING")