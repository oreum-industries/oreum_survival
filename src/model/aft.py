# src.model.aft.py
# copyright 2024 Oreum OÜ
"""Bayesian Survival Modelling: Accelerated Failure Time Models"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from oreum_core import model_pymc as mt
from pymc.distributions.dist_math import check_parameters

__all__ = ["WeibullRegression", "GompertzRegression", "GompertzRegressionAlt"]


class WeibullRegression(mt.BasePYMCModel):
    """Simple Weibull model incl censoring, unpooled static coeffs
    Core components used from oreum_core.model.BasePYMCModel
    As used by 200_AFT_Weibull.ipynb
    """

    name = "weibull_regression"
    version = "1.0.1"

    def __init__(
        self, obs: pd.DataFrame, fts_en: list, factor_map: dict, *args, **kwargs
    ):
        """Expects:
        obs: dfx dataframe for observations, arranged as: Y ~ X, aka
            pd.concat((dfx_en, dfx_exs), axis=1)
        fts_en: fts_en_t as defined during dataprep expect [t, d]
        factor_map: from Transformer.factor_map
        """
        super().__init__(*args, **kwargs)
        self.obs_nm = kwargs.pop("obs_nm", "obs")
        self.model = None
        self.rng = np.random.default_rng(seed=self.rsd)
        self.sample_kws.update(
            dict(
                target_accept=0.9,  # set higher to avoid divergences
                init="adapt_diag",  # avoid using jitter: sensitive startpos
                tune=4000,  # tune run longer
            )
        )
        # set obs, fts, and do data validity checks (None needed here)
        self.obs = obs.copy()
        self.fts_en = fts_en

        # set immutable coords indep of dataset(a | b) i.e. ft & var names
        # start with factors
        self.coords = {k: list(d.keys()) for k, d in factor_map.items()}
        # add on the rest
        self.coords.update(dict(x_nm=self.obs.drop(fts_en, axis=1).columns.values))

    def _build(self):
        """Build & return self.model, also set mutable coords i.e. obs_ids
        Aim the expected values to a reasonable range t approx in [0, 1000]
        """
        self.coords_m = dict(oid=self.obs.index.values)  # (i, )
        t_r = self.obs[self.fts_en[0]].values  # (i, )
        d_r = self.obs[self.fts_en[1]].values  # (i, )
        x_r = self.obs[self.coords["x_nm"]].values  # (i, x)

        with pm.Model(coords=self.coords, coords_mutable=self.coords_m) as self.model:
            # 0. create MutableData containers for obs (Y, X)
            t = pm.MutableData("t", t_r, dims="oid")  # (i, )
            d = pm.MutableData("d", d_r, dims="oid")  # (i, )
            x = pm.MutableData("x", x_r, dims=("oid", "x_nm"))  # (i, x)

            # 1. define priors and hyperpriors on alpha
            b_s = pm.Gamma("beta_s", alpha=1, beta=2)  # ( )
            # NOTE divergences when mu=0, so set 1
            b = pm.Normal("beta", mu=0.2, sigma=b_s, dims="x_nm")  # (x, )
            a = pm.Deterministic("alpha", pt.exp(pt.dot(x, b.T)), dims="oid")  # (i, )
            g = pm.Gamma("gamma", alpha=1, beta=200)  # E ~ a / b = 0.005

            # 2. define CustomDist log-likelihood incl. event censoring
            def _logp(
                t: pt.TensorVariable,
                alpha: pt.TensorVariable,
                gamma: pt.TensorVariable,
                d: pt.TensorVariable,
            ) -> pt.TensorVariable:
                """$log L(\pi)$
                NOTE:
                  + log-likelihood borrowed from pymc.Weibull.logp (incl. useful
                    switch and check_parameters), modified to incorporate d
                  + IMPORTANT do NOT sum logL to a scalar, because pymc
                    maintains the logL per observation all the way through into
                    idata.log_likelihood. i.e. we need to maintain the original
                    dimensions of y & yhat so that e.g. LOO-PIT can work later
                """
                res = d * (
                    pt.log(alpha) + pt.log(gamma) + (alpha - 1.0) * pt.log(gamma * t)
                ) - pt.power(gamma * t, alpha)
                res = pt.switch(pt.ge(t, 0.0), res, -np.inf)
                return check_parameters(
                    res, alpha > 0, gamma > 0, msg="alpha > 0, gamma > 0"
                )

            def _random(
                alpha: np.ndarray | float,
                gamma: np.ndarray | float,
                d: pt.TensorVariable,  # positional arg unused
                rng: np.random.Generator = None,
                size: tuple[int] = None,
            ) -> np.ndarray | float:
                """Create random durations: pass random uniform through the
                inverted Survival curve:
                where:  $S = \exp ( - (\gamma * t))$
                and:    $u \sim Uniform(0, 1)$
                so:     $t = S^{-1}(u) \sim 1/\gamma * (− \log u )^(1/\alpha)$
                """
                u = np.maximum(rng.uniform(size=size), 1e-12)  # avoid log(0)
                t = (1 / gamma) * np.power(-np.log(u), 1 / alpha)
                return np.round(t)  # TODO: quantize to obs unit reasonable?

            _ = pm.CustomDist(
                "that", a, g, d, logp=_logp, random=_random, observed=t, dims="oid"
            )

            # create survival function for convenience
            _ = pm.Deterministic("shat", pt.exp(-pt.power(g * t, a)), dims="oid")

        self.rvs_prior = ["gamma", "beta_s", "beta"]
        self.rvs_ppc = ["that", "shat"]
        self.rvs_det = ["alpha"]

        return self.model


class GompertzRegression(mt.BasePYMCModel):
    """Simple Gompertz model incl censoring, unpooled static coeffs
    Core components used from oreum_core.model.BasePYMCModel
    As used by 201_AFT_Gompertz.ipynb
    """

    name = "gompertz_regression"
    version = "1.0.0"

    def __init__(
        self, obs: pd.DataFrame, fts_en: list, factor_map: dict, *args, **kwargs
    ):
        """Expects:
        obs: dfx dataframe for observations, arranged as: Y ~ X, aka
            pd.concat((dfx_en, dfx_exs), axis=1)
        fts_en: fts_en_t as defined during dataprep expect [y, e]
        factor_map: from Transformer.factor_map
        """
        super().__init__(*args, **kwargs)
        self.obs_nm = kwargs.pop("obs_nm", "obs")
        self.model = None
        self.rng = np.random.default_rng(seed=self.rsd)
        self.sample_kws.update(
            dict(
                target_accept=0.85,  # set higher to avoid divergences
                init="adapt_diag",  # avoid using jitter: sensitive startpos
                tune=2000,  # tune run longer
            )
        )
        # set obs, fts, and do data validity checks (None needed here)
        self.obs = obs.copy()
        self.fts_en = fts_en

        # set immutable coords indep of dataset(a | b) i.e. ft & var names
        # start with factors
        self.coords = {k: list(d.keys()) for k, d in factor_map.items()}
        # add on the rest
        self.coords.update(dict(x_nm=self.obs.drop(fts_en, axis=1).columns.values))

    def _build(self):
        """Build & return self.model, also set mutable coords i.e. obs_ids
        Aim the expected values to a reasonable range t approx in [0, 1000]
        """
        self.coords_m = dict(oid=self.obs.index.values)  # (i, )
        t_r = self.obs[self.fts_en[0]].values  # (i, )
        d_r = self.obs[self.fts_en[1]].values  # (i, )
        x_r = self.obs[self.coords["x_nm"]].values  # (i, x)

        with pm.Model(coords=self.coords, coords_mutable=self.coords_m) as self.model:
            # 0. create MutableData containers for obs (Y, X)
            t = pm.MutableData("t", t_r, dims="oid")  # (i, )
            d = pm.MutableData("d", d_r, dims="oid")  # (i, )
            x = pm.MutableData("x", x_r, dims=("oid", "x_nm"))  # (i, x)

            # 1. define priors and hyperpriors on eta
            b_s = pm.InverseGamma("beta_s", alpha=5, beta=1)  # ( )
            b = pm.Normal("beta", mu=-3, sigma=b_s, dims="x_nm")  # (x, )
            e = pm.Deterministic("eta", pt.exp(pt.dot(x, b.T)), dims="oid")  # (i, )
            g = pm.InverseGamma("gamma", alpha=11, beta=1)

            # 2. define CustomDist log-likelihood incl. event censoring
            def _logp(
                t: pt.TensorVariable,
                eta: pt.TensorVariable,
                gamma: pt.TensorVariable,
                d: pt.TensorVariable,
            ) -> pt.TensorVariable:
                """$log L(\pi)$
                NOTE:
                + Use of exp1m borrowed from scipy.stats.gompertz_gen._pdf
                + IMPORTANT do NOT sum logL to a scalar, because pymc
                maintains the logL per observation all the way through into
                idata.log_likelihood. i.e. we need to maintain the original
                dimensions of y & yhat so that e.g. LOO-PIT can work later
                """
                res = d * (pt.log(eta) + pt.log(gamma) + gamma * t) - eta * pt.expm1(
                    gamma * t
                )
                res = pt.switch(pt.ge(t, 0.0), res, -np.inf)
                return check_parameters(
                    res, eta > 0, gamma > 0, msg="eta > 0, gamma > 0"
                )

            def _random(
                eta: np.ndarray | float,
                gamma: np.ndarray | float,
                d: pt.TensorVariable,  # positional arg unused
                rng: np.random.Generator = None,
                size: tuple[int] = None,
            ) -> np.ndarray | float:
                """Create random durations: pass random uniform through the
                inverted Survival curve:
                    where:  $S = \exp ( -\eta * ( \exp (\gamma * t) - 1))$,
                    and:    $u \sim Uniform(0, 1)$
                    so:     $t = S^{-1}(u) \sim 1/\gamma \log (1 - \log(u) / \eta)$
                NOTE does not create censored observations
                def _isf(self, p, c):
                Use of log1p borrowed from scipy.stats.gompertz_gen._isf
                """
                u = np.maximum(rng.uniform(size=size), 1e-12)  # avoid log(0)
                t = (1 / gamma) * np.log1p(-np.log(1 - u) / eta)
                return np.round(t)  # TODO: quantize to obs unit reasonable?

            _ = pm.CustomDist(
                "that", e, g, d, logp=_logp, random=_random, observed=t, dims="oid"
            )

            # create survival function for convenience
            _ = pm.Deterministic("shat", pt.exp(-e * pt.expm1(g * t)), dims="oid")

        self.rvs_prior = ["gamma", "beta_s", "beta"]
        self.rvs_ppc = ["that", "shat"]
        self.rvs_det = ["eta"]

        return self.model


class GompertzRegressionAlt(mt.BasePYMCModel):
    """Gompertz model with alternative parameterisation (modal, M)
    incl censoring, unpooled static coeffs Core components used from
    oreum_core.model.BasePYMCModel
    As used by 202_AFT_GompertzAlt.ipynb

    """

    name = "gompertz_regression_alt"
    version = "0.1.0"

    def __init__(
        self, obs: pd.DataFrame, fts_en: list, factor_map: dict, *args, **kwargs
    ):
        """Expects:
        obs: dfx dataframe for observations, arranged as: Y ~ X, aka
            pd.concat((dfx_en, dfx_exs), axis=1)
        fts_en: fts_en_t as defined during dataprep expect [y, e]
        factor_map: from Transformer.factor_map
        """
        super().__init__(*args, **kwargs)
        self.obs_nm = kwargs.pop("obs_nm", "obs")
        self.model = None
        self.rng = np.random.default_rng(seed=self.rsd)
        self.sample_kws.update(
            dict(
                target_accept=0.85,  # set higher to avoid divergences
                init="adapt_diag",  # avoid using jitter: sensitive startpos
                tune=2000,  # tune run longer
            )
        )
        # set obs, fts, and do data validity checks (None needed here)
        self.obs = obs.copy()
        self.fts_en = fts_en

        # set immutable coords indep of dataset(a | b) i.e. ft & var names
        # start with factors
        self.coords = {k: list(d.keys()) for k, d in factor_map.items()}
        # add on the rest
        self.coords.update(dict(x_nm=self.obs.drop(fts_en, axis=1).columns.values))

    def _build(self):
        """Build & return self.model, also set mutable coords i.e. obs_ids
        Aim the expected values to a reasonable range t approx in [0, 1000]
        """
        self.coords_m = dict(oid=self.obs.index.values)  # (i, )
        t_r = self.obs[self.fts_en[0]].values  # (i, )
        d_r = self.obs[self.fts_en[1]].values  # (i, )
        x_r = self.obs[self.coords["x_nm"]].values  # (i, x)

        with pm.Model(coords=self.coords, coords_mutable=self.coords_m) as self.model:
            # 0. create MutableData containers for obs (Y, X)
            t = pm.MutableData("t", t_r, dims="oid")  # (i, )
            d = pm.MutableData("d", d_r, dims="oid")  # (i, )
            x = pm.MutableData("x", x_r, dims=("oid", "x_nm"))  # (i, x)

            # 1. define priors and hyperpriors on M
            b_s = pm.InverseGamma("beta_s", alpha=5, beta=1)  # ( )
            b = pm.Normal("beta", mu=2, sigma=b_s, dims="x_nm")  # (x, )
            m = pm.Deterministic("m", pt.exp(pt.dot(x, b.T)), dims="oid")  # (i, )
            g = pm.InverseGamma("gamma", alpha=11, beta=1)

            # 2. define CustomDist log-likelihood incl. event censoring
            def _logp(
                t: pt.TensorVariable,
                m: pt.TensorVariable,
                g: pt.TensorVariable,
                d: pt.TensorVariable,
            ) -> pt.TensorVariable:
                """$log L(\pi)$
                NOTE:
                + IMPORTANT do NOT sum logL to a scalar, because pymc
                maintains the logL per observation all the way through into
                idata.log_likelihood. i.e. we need to maintain the original
                dimensions of y & yhat so that e.g. LOO-PIT can work later
                """
                res = (
                    d * (pt.log(g) + g * (t - m)) + pt.exp(-g * m) - pt.exp(g * (t - m))
                )
                res = pt.switch(pt.ge(t, 0.0), res, -np.inf)
                return check_parameters(res, m > 0, g > 0, msg="m > 0, gamma > 0")

            def _random(
                m: np.ndarray | float,
                g: np.ndarray | float,
                d: pt.TensorVariable,  # positional arg unused
                rng: np.random.Generator = None,
                size: tuple[int] = None,
            ) -> np.ndarray | float:
                """Create random durations: pass random uniform through the
                inverted Survival curve:
                    where:  $S = \exp( \exp(-\gamma * M) - \exp(\gamma * (t - M)) )$
                    and:    $u \sim Uniform(0, 1)$
                    so:     $t = S^{-1}(u) \sim 1/\gamma * \log(1 - (\log(1 - u) / \exp(-\gamma * M)))$
                NOTE does not create censored observations
                def _isf(self, p, c):
                Use of log1p borrowed from scipy.stats.gompertz_gen._isf
                """
                u = np.maximum(rng.uniform(size=size), 1e-12)  # avoid log(0)
                t = 1 / g * np.log1p(-np.log1p(-u) / np.exp(-g * m))
                return np.round(t)  # TODO: quantize to obs unit reasonable?

            _ = pm.CustomDist(
                "that", m, g, d, logp=_logp, random=_random, observed=t, dims="oid"
            )

            # create survival function for convenience
            _ = pm.Deterministic(
                "shat", pt.exp(pt.exp(-g * m) - pt.exp(g * (t - m))), dims="oid"
            )

        self.rvs_prior = ["gamma", "beta_s", "beta"]
        self.rvs_ppc = ["that", "shat"]
        self.rvs_det = ["m"]

        return self.model
