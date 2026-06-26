import torch


def get_free_vram_bytes(device: torch.device) -> int:
    """Return the amount of free VRAM in bytes on the given device."""
    if device.type != "cuda":
        raise ValueError(
            f"get_free_vram_bytes() expects a CUDA device, got '{device.type}'. "
            "Use batch_size=None for CPU runs."
        )
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return free_bytes


def estimate_batch_size(
    n_components: int,
    n_features: int,
    device: torch.device,
    covariance_type: str = "diag",
    safety_factor: float = 0.5,
    dtype: torch.dtype = torch.float32,
) -> int:
    """Estimate a conservative safe batch size for the E-step.

    The estimate depends on covariance_type because the dominant intermediate
    tensors differ a lot between diag/full/spherical.
    """
    if device.type != "cuda":
        raise ValueError(
            f"estimate_batch_size() expects a CUDA device, got '{device.type}'."
        )

    bytes_per_element = torch.finfo(dtype).bits // 8
    free_bytes = get_free_vram_bytes(device)
    usable_bytes = int(free_bytes * safety_factor)

    K = n_components
    D = n_features

    if covariance_type == "diag":
        # Conservative model:
        # - X_batch: (B, D)
        # - outputs: roughly (B, K)
        # - matmul intermediates and temp buffers
        # The previous estimate O(K + D) was too optimistic.
        bytes_per_sample = bytes_per_element * (
            D + 3 * K + 4 * K * D
        )

    elif covariance_type == "full":
        # Full covariance tends to be much more memory hungry due to
        # (B, K, D), (B, K, D, 1), matmul temporaries, etc.
        bytes_per_sample = bytes_per_element * (
            D + 4 * K * D + 2 * K * D * D
        )

    elif covariance_type == "spherical":
        bytes_per_sample = bytes_per_element * (
            D + 2 * K + 2 * K * D
        )

    else:
        raise ValueError(
            f"Unsupported covariance_type: {covariance_type!r}. "
            "Expected 'diag', 'full', or 'spherical'."
        )

    batch_size = usable_bytes // max(1, bytes_per_sample)
    return max(1, batch_size)


def resolve_batch_size(
    batch_size,
    n_samples: int,
    n_components: int,
    n_features: int,
    device: torch.device,
    covariance_type: str = "diag",
    dtype: torch.dtype = torch.float32,
) -> int:
    """Resolve the batch_size parameter to a concrete integer."""
    if batch_size is None:
        return n_samples

    if batch_size == "auto":
        if device.type != "cuda":
            return n_samples

        estimated = estimate_batch_size(
            n_components=n_components,
            n_features=n_features,
            device=device,
            covariance_type=covariance_type,
            dtype=dtype,
        )
        return min(estimated, n_samples)

    if isinstance(batch_size, int) and batch_size > 0:
        return min(batch_size, n_samples)

    raise ValueError(
        f"batch_size must be None, 'auto', or a positive integer. Got: {batch_size!r}"
    )


def iter_batches(X: torch.Tensor, batch_size: int):
    """Yield successive slices of X along the first dimension."""
    n_samples = X.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield X[start:end], start, end


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        free = get_free_vram_bytes(device)
        print(f"Free VRAM: {free / 1024**3:.2f} GB")

        for cov in ["diag", "full", "spherical"]:
            print(f"\nCovariance type: {cov}")
            for n_comp, n_feat in [(10, 20), (50, 100), (100, 200)]:
                bs = estimate_batch_size(
                    n_components=n_comp,
                    n_features=n_feat,
                    device=device,
                    covariance_type=cov,
                )
                print(f"  n_components={n_comp}, n_features={n_feat} -> batch_size={bs:,}")

    resolved = resolve_batch_size("auto", 1_000_000, 50, 100, device, covariance_type="diag")
    print(f"\nresolved batch_size (auto, diag): {resolved:,}")

    resolved = resolve_batch_size(None, 1_000_000, 50, 100, device, covariance_type="diag")
    print(f"resolved batch_size (None): {resolved:,}")

    resolved = resolve_batch_size(32768, 1_000_000, 50, 100, device, covariance_type="diag")
    print(f"resolved batch_size (32768): {resolved:,}")