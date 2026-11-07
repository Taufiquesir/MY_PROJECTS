"""Soft matting refinement via the Levin closed-form matting Laplacian."""
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def matting_laplacian(img, eps=1e-7, r=1):
    """Sparse matting Laplacian L (N x N) for HxWx3 img in [0,1], window (2r+1)^2."""
    h, w, c = img.shape
    wsz = (2 * r + 1) ** 2
    idx = np.arange(h * w).reshape(h, w)
    # window pixel indices for every window centre (interior only)
    ys, xs = np.meshgrid(np.arange(r, h - r), np.arange(r, w - r), indexing="ij")
    ys, xs = ys.ravel(), xs.ravel()
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    win_idx = np.stack([idx[ys + dy, xs + dx] for dy, dx in offs], axis=1)   # M x wsz
    win_pix = img.reshape(-1, c)[win_idx]                                     # M x wsz x c
    mu = win_pix.mean(axis=1, keepdims=True)
    d = win_pix - mu
    cov = np.einsum("mik,mil->mkl", d, d) / wsz
    inv = np.linalg.inv(cov + (eps / wsz) * np.eye(c))
    # L_ij = delta_ij - 1/wsz * (1 + d_i^T inv d_j)
    t = np.einsum("mik,mkl,mjl->mij", d, inv, d)
    vals = np.eye(wsz)[None] - (1.0 + t) / wsz                               # M x wsz x wsz
    rows = np.repeat(win_idx[:, :, None], wsz, axis=2).ravel()
    cols = np.repeat(win_idx[:, None, :], wsz, axis=1).ravel()
    return sp.coo_matrix((vals.ravel(), (rows, cols)), shape=(h * w, h * w)).tocsr()


_MAXITER = 20000


def soft_matting(img, t_raw, eps=1e-7, lam=1e-4, r=1):
    """Solve (L + lam*I) t = lam * t_raw."""
    h, w = t_raw.shape
    L = matting_laplacian(img, eps, r)
    A = (L + lam * sp.identity(h * w, format="csr")).tocsc()
    b, x0 = lam * t_raw.ravel(), t_raw.ravel()
    try:                                   # scipy >= 1.12 renamed tol -> rtol
        t, info = spla.cg(A, b, x0=x0, rtol=1e-6, atol=0, maxiter=_MAXITER)
    except TypeError:
        t, info = spla.cg(A, b, x0=x0, tol=1e-6, atol=0, maxiter=_MAXITER)
    if info != 0:       # ~2200 iterations are needed at 620x460; a lower cap stops unconverged
        warnings.warn(f"soft matting CG did not converge (info={info}); result is solver-path dependent")
    return np.clip(t.reshape(h, w), 0.0, 1.0)
