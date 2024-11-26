# src.model.exponential.py
# copyright 2024 Oreum OÜ
"""Bayesian Survival Modelling: Exponential Models
Technically speaking, the CoxPH1/2/3 models should also be in here because
they're also classed as "Exponential" models, and were originally in here
alongside CoxPH0, but I've moved those to separate file `piecewise` for ease
of later / wider understanding and to match the notebook naming / grouping.
Also technically, the ExponentialRegression model is an "AFT" model, but
also keeping it in here for ease of understanding.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from oreum_core import model_pymc as mt

__all__ = ["ExponentialUnivariate", "ExponentialRegression", "CoxPH0"]


class ExponentialUnivariate(mt.BasePYMCModel):
    """Simple Exponential model incl censoring, fully pooled (no covariates)
    Core components used from oreum_core.model.BasePYMCModel
    As used by 100_Exponential_Univariate.ipynb
    """

    name = "exponential_univariate"
    version = "1.0.2"

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
            dict(target_accept=0.80)  # set higher to avoid divergences
        )
        # set obs, fts, and do data validity checks (None needed here)
        self.obs = obs.copy()
        self.fts_en = fts_en

        # set immutable coords indep of dataset(a | b) i.e. ft & var names
        # start with factors
        self.coords = {k: list(d.keys()) for k, d in factor_map.items()}
        # add on the rest
        # not needed

    def _build(self):
        """Build & return self.model, also set mutable coords i.e. obs_ids
        We set the expected value for the `gamma` prior based on the KM measure
        (Section 1.2.2) of median survival @ t=113:
        $S = 0.5 \sim exp(-\gamma 113)$ so
        $ \gamma \sim - log(0.5) / 113 \sim 0.003$
        """
        self.coords_m = dict(oid=self.obs.index.values)  # (44, )
        t_r = self.obs[self.fts_en[0]].values  # (44,)
        d_r = self.obs[self.fts_en[1]].values  # (44,)

        with pm.Model(coords=self.coords, coords_mutable=self.coords_m) as self.model:
            # 0. create MutableData containers for obs (Y, X)
            t = pm.MutableData("t", t_r, dims="oid")
            d = pm.MutableData("d", d_r, dims="oid")

            # 1. define freeRVs
            # NOTE just for demo (not actually needed here), get opt_params
            # Set init_guess to create Eval, gamma E ~ a / b 0.005
            # Set lower & upper to center on the Expected value, allow some span
            # opt_params = pm.find_constrained_prior(pm.Gamma, lower=1/5*mn,
            # upper=39/5*mn, mass=0.8, init_guess=dict(alpha=5, beta=1000))
            # gamma = pm.Gamma.dist(**opt_params, size=100).eval()

            # NOTE: will set manually and wider
            g = pm.Gamma("gamma", alpha=2, beta=400)  # E ~ a / b = 0.005

            # 2. define CustomDist log-likelihood incl. event censoring
            def _logp(
                t: pt.TensorVariable, gamma: pt.TensorVariable, d: pt.TensorVariable
            ) -> pt.TensorVariable:
                """$log L(\pi) \sim \sum_{i} d_{i} log \gamma - \gamma t_{i}$
                NOTE: IMPORTANT do NOT sum logL to a scalar, because pymc
                maintains the logL per observation all the way through into
                idata.log_likelihood. i.e. we need to maintain the original
                dimensions of y & yhat so that e.g. LOO-PIT can work later"""
                return (d * pt.log(gamma)) - (gamma * t)

            def _random(
                gamma: np.ndarray | float,
                d: pt.TensorVariable,  # positional unused
                rng: np.random.Generator = None,
                size: tuple[int] = None,
            ) -> np.ndarray | float:
                """Create random durations: pass random uniform through the
                inverted Survival curve:
                    where:  $S = \exp ( - (\gamma * t))$
                    and:    $u \sim Uniform(0, 1)$
                    so:     $t \sim S^{-1}(u) \sim 1/\gamma * (− \log u )$

                NOTE cannot create censored observations
                """
                u = np.maximum(rng.uniform(size=size), 1e-12)  # avoid log(0)
                t = (1 / gamma) * -np.log(u)
                return np.round(t, 0)  # TODO: quantize to obs unit reasonable?

            _ = pm.CustomDist(
                "that", g, d, logp=_logp, random=_random, observed=t, dims="oid"
            )

            _ = pm.Deterministic("shat", pt.exp(-g * t), dims="oid")

        self.rvs_g = ["gamma"]
        self.rvs_ppc = ["that", "shat"]

        return self.model


class ExponentialRegression(mt.BasePYMCModel):
    """Simple Exponential model incl censoring, unpooled static coeffs
    Core components used from oreum_core.model.BasePYMCModel
    As used by 101_Exponential_Regression.ipynb
    """

    name = "exponential_regression"
    version = "1.0.2"

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
            dict(target_accept=0.80)  # set higher to avoid divergences
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
        Aim the expected value for the `gamma` prior based on the KM measure
        (Section 1.2.2) of median survival @ t=113:
        $S = 0.5 \sim exp(-\gamma 113)$ so
        $ \gamma \sim - log(0.5) / 113 \sim 0.003$
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

            # 1. define freeRVs linear model
            # NOTE The model is very sensitive to gamma and needs to be able to
            # go very small. So we'll aim Egamma ~ 0.005
            # meaning E(beta.t) ~ np.log(0.005) ~ -2.3, so put betas at E ~ -1
            # with a wide sigma = 2, then 2sd (HDI94) in [-6, 2], good range
            # NOTE maintain gamma as a Deterministic for convenience
            beta = pm.Normal("beta", mu=-2, sigma=2, dims="x_nm")  # (x, )
            g = pm.Deterministic("gamma", pt.exp(pt.dot(x, beta.T)), dims="oid")

            # 2. define CustomDist log-likelihood incl. event censoring
            def _logp(
                t: pt.TensorVariable, gamma: pt.TensorVariable, d: pt.TensorVariable
            ) -> pt.TensorVariable:
                """$log L(\pi) \sim \sum_{i} d_{i} log \gamma - \gamma t_{i}$
                NOTE: IMPORTANT do NOT sum logL to a scalar, because pymc
                maintains the logL per observation all the way through into
                idata.log_likelihood. i.e. we need to maintain the original
                dimensions of y & yhat so that e.g. LOO-PIT can work later
                """
                return (d * pt.log(gamma)) - (gamma * t)

            def _random(
                gamma: np.ndarray | float,
                d: pt.TensorVariable,  # positional unused
                rng: np.random.Generator = None,
                size: tuple[int] = None,
            ) -> np.ndarray | float:
                """Create random durations: pass random uniform through the
                inverted Survival curve:
                where:  $S = \exp ( - (\gamma * t))$
                and:    $u \sim Uniform(0, 1)$
                so:     $t \sim S^{-1}(u) \sim 1/\gamma * (− \log u )$
                """
                u = np.maximum(rng.uniform(size=size), 1e-12)  # avoid log(0)
                t = -np.log(u) / gamma
                return np.round(t)  # TODO: quantize to obs unit reasonable?

            _ = pm.CustomDist(
                "that", g, d, logp=_logp, random=_random, observed=t, dims="oid"
            )

            # create survival function for convenience
            _ = pm.Deterministic("shat", pt.exp(-g * t), dims="oid")

        self.rvs_b = ["beta"]
        self.rvs_ppc = ["that", "shat"]
        self.rvs_det = ["gamma"]

        return self.model


class CoxPH0(mt.BasePYMCModel):
    """Cox Proportional Hazards Piecewise Exponential model incl. censoring,
    CoxPH0 has an unpooled baseline hazard and unpooled static coeffs
    Core components used from oreum_core.model.BasePYMCModel
    As used by 102_Exponential_CoxPH0.ipynb
    """

    name = "coxph0"
    version = "1.2.0"

    def __init__(
        self,
        obs: pd.DataFrame,
        fts_en: list,
        factor_map: dict,
        bin_width: int,
        *args,
        **kwargs,
    ):
        """Expects:
        obs: dfx dataframe for observations, arranged as: Y ~ X, aka
            pd.concat((dfx_en, dfx_exs), axis=1)
            NOTE: for compatibility with upstream data transformation we allow
            dfx_exs to contain an intercept column called 'intercept'
            but this will be ignored because CoxPH already has an intercept
            via the lambda0 variable and to keep 'intercept' would lead to
            identifiability issues
        fts_en: fts_en_t as defined during dataprep expect [duration, event]
        factor_map: from Transformer.factor_map
        # [Future dev] bin_edges: the inclusive edges of the intervals (bins) j
        bin_width: equal width bins, less good than bin_edges, but ok for now
        """
        super().__init__(*args, **kwargs)
        self.obs_nm = kwargs.pop("obs_nm", "obs")
        self.model = None
        self.rng = np.random.default_rng(seed=self.rsd)
        self.sample_kws.update(
            dict(target_accept=0.80)  # set higher to avoid divergences
        )
        # set obs, fts, and do data validity checks
        self.obs = obs.copy()
        self.fts_en = fts_en  # note we require [duration, event]
        self.bin_width = bin_width
        m = 0 if self.bin_width == 1 else 1
        self.bin_edges = self.bin_width * np.arange(
            (self.obs["duration"].max() // self.bin_width) + m + 1
        )

        # set immutable coords indep of dataset(a | b) i.e. ft & var names
        # start with factors
        self.coords = {k: list(d.keys()) for k, d in factor_map.items()}
        # add on the rest
        self.coords.update(
            dict(
                x_nm=self.obs.drop(self.fts_en + ["intercept"], axis=1).columns.values,
                j_nm=[
                    f"{v0:.0f}<t<={v1:.0f}"
                    for v0, v1 in zip(self.bin_edges, self.bin_edges[1:], strict=False)
                ],
            )
        )

    def _build(self):
        """Build & return self.model, also set mutable coords i.e. obs_ids"""
        self.coords_m = dict(oid=self.obs.index.values)  # (i,)
        t_r = self.obs[self.fts_en[0]].values  # (i,)
        d_r = self.obs[self.fts_en[1]].values  # (i,)
        x_r = self.obs[self.coords["x_nm"]].values  # (i, 1)

        # create intervals j
        j_r = np.digitize(t_r, self.bin_edges[1:], right=True)

        # create t_ij
        ## step 1: create the indicators
        # corrected Austin's broadcasting to put False in the correct bin
        tid = np.full((len(t_r), j_r.max() + 2), True, dtype=np.bool_)
        np.put_along_axis(tid, j_r.reshape(-1, 1) + 1, False, axis=1)
        tid = tid[:, :-1]
        tid = np.minimum.accumulate(tid, axis=1)
        ## step 2: Step 2: Fill with j interval value
        ## and overwrite the final observed interval with the "fulled" remainder
        ## $t = \tau_{j-1}$ on the ragged edge cells
        rem_full = np.mod(t_r, self.bin_width)
        rem_full[rem_full == 0] = self.bin_width
        t_ij_r = tid * self.bin_width
        np.put_along_axis(t_ij_r, j_r.reshape(-1, 1), rem_full.reshape(-1, 1), axis=1)

        # create d_ij
        d_ij_r = np.full_like(t_ij_r, False, dtype=np.bool_)
        np.put_along_axis(d_ij_r, j_r.reshape(-1, 1), d_r.reshape(-1, 1), axis=1)

        with pm.Model(coords=self.coords, coords_mutable=self.coords_m) as self.model:
            # 0. create MutableData containers for obs (Y, X)
            t_ij = pm.MutableData("t_ij", t_ij_r, dims=("oid", "j_nm"))  # (i, j)
            d_ij = pm.MutableData("d_ij", d_ij_r, dims=("oid", "j_nm"))  # (i, j)
            x = pm.MutableData("x", x_r, dims=("oid", "x_nm"))  # (i, 1)

            # 1. define baseline hazard as unpooled Gamma
            l_j = pm.Gamma("lambda_j", alpha=2, beta=400, dims="j_nm")  # E ~ 0.005

            # 2. define covariate RVs
            b = pm.Normal("beta", mu=0, sigma=2, dims="x_nm")

            # 3. build up mu
            l_i_ = pt.exp(pt.dot(x, b.T))  # (i, )
            l_ij_ = pt.outer(l_i_, l_j)  # (i, j)
            mu_ = t_ij * l_ij_ + 1e-12  # avoid numerical errors near 0 (i, j)

            # 4. define likelihood
            _ = pm.Poisson("dhat_ij", mu=mu_, observed=d_ij, dims=("oid", "j_nm"))

            # create predicted survival function for convenience
            _ = pm.Deterministic(
                "shat_ij",
                pt.exp(-pt.cumsum(t_ij * l_ij_, axis=1)),
                dims=("oid", "j_nm"),
            )

        self.rvs_lam = ["lambda_j"]
        self.rvs_b = ["beta"]
        self.rvs_ppc = ["dhat_ij", "shat_ij"]

        return self.model
