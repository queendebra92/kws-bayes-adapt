"""Gaussian Bayes classifier with a minimum-risk reject option.

Class-conditional densities p(x | w_i) = N(mu_i, Sigma_i), priors from the
training labels, decision a*(x) = argmax_i [log p(x | w_i) + log P(w_i)].
Sigma_i is regularised: Sigma_i + eps * (tr(Sigma_i) / d) * I, with eps tuned
on validation data only.

Reject option (Chow, 1970): with 0-1 loss and rejection cost lambda_r < 1,
reject iff max_i P(w_i | x) < 1 - lambda_r.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.special import logsumexp

REJECT = -1


class GaussianBayes:
    def __init__(self, eps: float = 1e-2):
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> "GaussianBayes":
        X = np.asarray(X, dtype=np.float64)
        self.n_classes = int(n_classes if n_classes is not None else y.max() + 1)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-8
        Z = (X - self.mean_) / self.scale_
        d = Z.shape[1]
        self.mu_ = np.zeros((self.n_classes, d))
        self.cov_ = np.zeros((self.n_classes, d, d))
        self.log_prior_ = np.full(self.n_classes, -np.inf)
        for c in range(self.n_classes):
            Zc = Z[y == c]
            if len(Zc) < 2:
                continue
            self.mu_[c] = Zc.mean(axis=0)
            S = np.atleast_2d(np.cov(Zc, rowvar=False))
            self.cov_[c] = S + self.eps * (np.trace(S) / d) * np.eye(d)
            self.log_prior_[c] = np.log(len(Zc) / len(Z))
        self._factor()
        return self

    def _factor(self) -> None:
        self._chol, self._logdet = {}, np.zeros(self.n_classes)
        for c in range(self.n_classes):
            if np.isfinite(self.log_prior_[c]):
                Lc = cholesky(self.cov_[c], lower=True)
                self._chol[c] = Lc
                self._logdet[c] = 2.0 * np.log(np.diag(Lc)).sum()

    def log_joint(self, X: np.ndarray) -> np.ndarray:
        """log p(x | w_i) + log P(w_i), shape [N, C]."""
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        d = Z.shape[1]
        out = np.full((len(Z), self.n_classes), -np.inf)
        for c, Lc in self._chol.items():
            diff = (Z - self.mu_[c]).T
            sol = cho_solve((Lc, True), diff)
            maha = np.einsum("ij,ij->j", diff, sol)
            out[:, c] = -0.5 * (maha + self._logdet[c] + d * np.log(2.0 * np.pi)) + self.log_prior_[c]
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lj = self.log_joint(X)
        return np.exp(lj - logsumexp(lj, axis=1, keepdims=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.log_joint(X).argmax(axis=1)

    def predict_reject(self, X: np.ndarray, lambda_r: float) -> np.ndarray:
        """Chow rule; returns class index or REJECT (-1)."""
        p = self.predict_proba(X)
        pred = p.argmax(axis=1)
        pred[p.max(axis=1) < 1.0 - lambda_r] = REJECT
        return pred


class Preproc:
    """Standardise, then optional PCA with whitening. Fit on training data only."""

    def __init__(self, pca_dim: int | None = None):
        self.pca_dim = pca_dim

    def fit(self, X: np.ndarray) -> "Preproc":
        from sklearn.decomposition import PCA
        X = np.asarray(X, dtype=np.float64)
        self.mean_, self.scale_ = X.mean(axis=0), X.std(axis=0) + 1e-8
        self.pca_ = None
        if self.pca_dim:
            self.pca_ = PCA(n_components=self.pca_dim, whiten=True, random_state=0)
            self.pca_.fit((X - self.mean_) / self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return self.pca_.transform(Z) if self.pca_ is not None else Z


class GMMBayes:
    """Bayes classifier whose class-conditional densities are Gaussian mixtures (EM).

    k[c] components for class c (full covariance, reg_covar added to the diagonal).
    Expects already-preprocessed (standardised / whitened) inputs.
    """

    def __init__(self, k, reg_covar: float = 1e-2, max_iter: int = 60, seed: int = 0):
        self.k, self.reg_covar, self.max_iter, self.seed = k, reg_covar, max_iter, seed

    def fit(self, Z: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> "GMMBayes":
        from sklearn.mixture import GaussianMixture
        self.n_classes = int(n_classes if n_classes is not None else y.max() + 1)
        self.gm_ = {}
        self.log_prior_ = np.full(self.n_classes, -np.inf)
        for c in range(self.n_classes):
            Zc = Z[y == c]
            if len(Zc) < 2:
                continue
            k = self.k[c] if not np.isscalar(self.k) else int(self.k)
            k = max(1, min(k, len(Zc) // 20 or 1))
            self.gm_[c] = GaussianMixture(
                n_components=k, covariance_type="full", reg_covar=self.reg_covar,
                max_iter=self.max_iter, random_state=self.seed, init_params="kmeans").fit(Zc)
            self.log_prior_[c] = np.log(len(Zc) / len(Z))
        return self

    def log_joint(self, Z: np.ndarray) -> np.ndarray:
        out = np.full((len(Z), self.n_classes), -np.inf)
        for c, g in self.gm_.items():
            out[:, c] = g.score_samples(Z) + self.log_prior_[c]
        return out

    def predict_proba(self, Z):
        lj = self.log_joint(Z)
        return np.exp(lj - logsumexp(lj, axis=1, keepdims=True))

    def predict(self, Z):
        return self.log_joint(Z).argmax(axis=1)

    def predict_reject(self, Z, lambda_r: float):
        p = self.predict_proba(Z)
        pred = p.argmax(axis=1)
        pred[p.max(axis=1) < 1.0 - lambda_r] = REJECT
        return pred


def adapt_em(model: "GaussianBayes", Zu: np.ndarray, tau: float = 16.0, n_iter: int = 10) -> "GaussianBayes":
    """Unsupervised MAP-EM adaptation of the class-conditional Gaussians.

    Zu: UNLABELED adaptation inputs in the same (preprocessed) space the model was fitted on.
    Class priors pi_c stay fixed at their clean-training values; only (mu_c, Sigma_c) adapt.

    E-step : gamma_ic = pi_c N(z_i | mu_c, Sigma_c) / sum_k pi_k N(z_i | mu_k, Sigma_k)
    M-step : n_c = sum_i gamma_ic,  m_c = sum_i gamma_ic z_i / n_c
             mu_c    = (tau mu0_c + n_c m_c) / (tau + n_c)
             Sigma_c = (tau Sigma0_c + sum_i gamma_ic (z_i - mu_c)(z_i - mu_c)^T
                        + tau (mu_c - mu0_c)(mu_c - mu0_c)^T) / (tau + n_c)
    (MAP estimate under a Normal-Wishart prior centred on the clean parameters, relevance tau;
    Gauvain & Lee 1994.) tau -> infinity returns the clean model; tau -> 0 is plain EM.
    """
    import copy
    Z = (np.asarray(Zu, dtype=np.float64) - model.mean_) / model.scale_
    d = Z.shape[1]
    new = copy.copy(model)
    new.mu_, new.cov_ = model.mu_.copy(), model.cov_.copy()
    live = [c for c in range(model.n_classes) if np.isfinite(model.log_prior_[c])]
    for _ in range(n_iter):
        new._factor()
        lj = new.log_joint(model.mean_ + Z * model.scale_)          # same space as the model expects
        lj -= logsumexp(lj, axis=1, keepdims=True)
        gamma = np.exp(lj)
        for c in live:
            g = gamma[:, c]
            nc = g.sum()
            if nc < 1e-8:
                new.mu_[c], new.cov_[c] = model.mu_[c], model.cov_[c]
                continue
            mc = (g[:, None] * Z).sum(axis=0) / nc
            mu = (tau * model.mu_[c] + nc * mc) / (tau + nc)
            R = Z - mu
            S = (R * g[:, None]).T @ R
            dm = (mu - model.mu_[c])[:, None]
            new.mu_[c] = mu
            new.cov_[c] = (tau * model.cov_[c] + S + tau * (dm @ dm.T)) / (tau + nc)
    new._factor()
    return new


class AffineAdapted:
    """A fitted model plus a SHARED per-dimension shift and scale applied to every input.

    In the model's own standardised space the input is mapped z -> (z - m) / s, which is the same as
    transforming all class means and covariances jointly (mu_c -> s*mu_c + m, Sigma_c -> S Sigma_c S).
    Only 2d parameters are estimated, from ALL unlabeled adaptation clips, with no class assignments,
    so it cannot re-purpose individual class densities (the failure mode of per-class EM).
    """

    def __init__(self, model, m: np.ndarray, s: np.ndarray):
        self.model, self.m, self.s = model, m, s
        self.n_classes = model.n_classes

    def _t(self, X):
        z = (np.asarray(X, dtype=np.float64) - self.model.mean_) / self.model.scale_
        return self.model.mean_ + ((z - self.m) / self.s) * self.model.scale_

    def transform_for_adaptation(self, X):
        return self._t(X)

    def log_joint(self, X):
        return self.model.log_joint(self._t(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._t(X))

    def predict(self, X):
        return self.model.predict(self._t(X))

    def predict_reject(self, X, lambda_r: float):
        return self.model.predict_reject(self._t(X), lambda_r)


def fit_affine(model: "GaussianBayes", Zu: np.ndarray, kappa: float = 20.0) -> AffineAdapted:
    """Estimate the shared shift m and scale s from unlabeled adaptation inputs Zu.

    In the model's standardised space the clean training marginal has mean 0 and variance 1 per
    dimension, so m, s are the adaptation set's mean and std, shrunk toward (0, 1) with weight kappa:
        m = n/(n+kappa) * mean_u,   s^2 = (n*var_u + kappa*1)/(n + kappa).
    With data that already matches training, (m, s) ~ (0, 1) and nothing changes.
    """
    Z = (np.asarray(Zu, dtype=np.float64) - model.mean_) / model.scale_
    n = len(Z)
    mean_u, var_u = Z.mean(axis=0), Z.var(axis=0)
    m = n / (n + kappa) * mean_u
    s = np.sqrt((n * var_u + kappa) / (n + kappa))
    return AffineAdapted(model, m, s)


def adapt_affine_then_em(model: "GaussianBayes", Zu: np.ndarray, kappa: float = 20.0,
                         tau: float = 1024.0, n_iter: int = 10) -> AffineAdapted:
    """Shared affine adaptation first, then a strongly anchored MAP-EM refinement of the class densities."""
    aff = fit_affine(model, Zu, kappa)
    em = adapt_em(model, aff.transform_for_adaptation(Zu), tau, n_iter)
    return AffineAdapted(em, aff.m, aff.s)
