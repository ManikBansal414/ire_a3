"""Seeded synthetic data for Part 4: Gaussian data with a chosen intrinsic
dimension inside a fixed 128-dimensional ambient space."""
import numpy as np

AMBIENT_DIM = 128
NOISE_VAR = 1e-3  # total variance of the isotropic noise (signal total variance = 1)


def subspace_gaussian(n_base, n_query, d_int, seed, dim=AMBIENT_DIM, noise_var=NOISE_VAR):
    """Base and query vectors drawn from the same distribution:

        x = B z + e,  z ~ N(0, I/d_int) in R^d_int,  e ~ N(0, I * noise_var/dim)

    B is a random orthonormal dim x d_int basis. Variance is split evenly over
    the d_int directions so E||Bz||^2 = 1 for every d_int: only the *shape*
    of the data changes with d_int, not its scale. d_int = dim gives a plain
    isotropic Gaussian.
    """
    rng = np.random.default_rng(seed)
    B, _ = np.linalg.qr(rng.standard_normal((dim, d_int)))   # dim x d_int, orthonormal columns

    def draw(n):
        out = np.empty((n, dim), dtype=np.float32)
        for s in range(0, n, 50_000):  # chunked to keep peak memory small
            m = min(50_000, n - s)
            z = rng.standard_normal((m, d_int)) / np.sqrt(d_int)
            e = rng.standard_normal((m, dim)) * np.sqrt(noise_var / dim)
            out[s:s + m] = z @ B.T + e
        return out

    return draw(n_base), draw(n_query)


def relative_contrast(xb, xq, nn_sq_dist, n_sample=2000, seed=0):
    """Per query: mean distance to random base points / distance to the nearest
    neighbour (He et al., 2012). Near 1 means the nearest neighbour is barely
    closer than a random point."""
    rng = np.random.default_rng(seed)
    s = xb[rng.choice(len(xb), n_sample, replace=False)].astype(np.float64)
    q = xq.astype(np.float64)
    d2 = (q ** 2).sum(1)[:, None] + (s ** 2).sum(1)[None] - 2 * q @ s.T
    mean_d = np.sqrt(np.maximum(d2, 0)).mean(1)
    return mean_d / np.sqrt(np.maximum(nn_sq_dist, 1e-30))


def lid_mle(knn_sq_dist):
    """Maximum-likelihood local intrinsic dimension from the k nearest-neighbour
    distances of each query (Levina & Bickel); returns per-query estimates."""
    d = np.sqrt(np.maximum(knn_sq_dist.astype(np.float64), 1e-30))
    k = d.shape[1]
    logs = np.log(d[:, -1:] / d[:, :-1]).sum(1) / (k - 1)
    return 1.0 / np.maximum(logs, 1e-12)
