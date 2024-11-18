# src.model.piecewise.py
# copyright 2024 Oreum OÜ
"""Bayesian Survival Modelling: Piecewise Models
    Technically speaking, these are also classed as "Exponential" models, and
    were originally in `exponential.py` alongside CoxPH0, but I've moved them
    to this separate file for ease of later / wider understanding and to match
    the notebook naming / grouping.
"""
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from oreum_core import model_pymc as mt

__all__ = ['CoxPH1', 'CoxPH2', 'CoxPH3']


class CoxPH1(mt.BasePYMCModel):
    """Cox Proportional Hazards Piecewise Exponential model incl. censoring,
    CoxPH1 has a partial-pooled baseline hazard and unpooled static coeffs
    Core components used from oreum_core.model.BasePYMCModel
    As used by 300_Piecewise_CoxPH1.ipynb

    NOTE: We use a non-centered parameterisation because the data is relatively
    sparse, thus the FreeRV parmeters will be less well-informed, and we want
    to allow greater space of sampling. For more info see
    [Hierarchical Modeling, Betancourt, Nov 2020, Section 3.2]
    (https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html),
    a worked example from
    [Thomas Weicki's blog](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/),
    and more dicussion of the practical impacts on the
    [pymc Discourse](https://discourse.pymc.io/t/non-centering-results-in-lower-effective-sample-size/5850)

    """

    name = 'coxph1'
    version = '1.1.0'

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
        self.obs_nm = kwargs.pop('obs_nm', 'obs')
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
            (self.obs['duration'].max() // self.bin_width) + m + 1
        )

        # set immutable coords indep of dataset(a | b) i.e. ft & var names
        # start with factors
        self.coords = {k: list(d.keys()) for k, d in factor_map.items()}
        # add on the rest
        self.coords.update(
            dict(
                x_nm=self.obs.drop(self.fts_en + ['intercept'], axis=1).columns.values,
                j_nm=[
                    f'{v0:.0f}<t<={v1:.0f}'
                    for v0, v1 in zip(self.bin_edges, self.bin_edges[1:])
                ],
            )
        )

    def _build(self):
        """Build & return self.model, also set mutable coords i.e. obs_ids"""
        self.coords_m = dict(oid=self.obs.index.values)  # (44,)
        t_r = self.obs[self.fts_en[0]].values  # (44,)
        d_r = self.obs[self.fts_en[1]].values  # (44,)
        x_r = self.obs[self.coords['x_nm']].values  # (44, 1)

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
            t_ij = pm.MutableData('t_ij', t_ij_r, dims=('oid', 'j_nm'))  # (44, j)
            d_ij = pm.MutableData('d_ij', d_ij_r, dims=('oid', 'j_nm'))  # (44, j)
            x = pm.MutableData('x', x_r, dims=('oid', 'x_nm'))  # (44, 1)

            # 1. define baseline hazard as log-transformed hierarchical Normal,
            # with non-centered parameterisation prior E(lj) ~ exp(-5) = 0.007
            l_j_m = pm.Normal('lambda_j_mu', mu=-5, sigma=1)
            l_j_m_o = pm.Normal('lambda_j_mu_offset', mu=0, sigma=1, dims='j_nm')
            l_j_s = pm.InverseGamma('lambda_j_sigma', alpha=101, beta=100)
            l_j_ = pt.exp(l_j_m + (l_j_m_o * l_j_s))
            l_j = pm.Deterministic('lambda_j', l_j_, dims='j_nm')

            # 2. define covariate RVs
            b = pm.Normal('beta', mu=0, sigma=1, dims='x_nm')

            # 3. build up mu
            l_i_ = pt.exp(pt.dot(x, b.T))  # (44, )
            l_ij_ = pt.outer(l_i_, l_j)  # (44, j)
            mu_ = t_ij * l_ij_ + 1e-12  # avoid numerical errors near 0 (44, j)

            # 4. define likelihood
            _ = pm.Poisson('dhat_ij', mu=mu_, observed=d_ij, dims=('oid', 'j_nm'))

            # create predicted survival function for convenience
            _ = pm.Deterministic(
                'shat_ij', pt.exp(-pt.cumsum(mu_, axis=1)), dims=('oid', 'j_nm')
            )

        self.rvs_lam = ['lambda_j_mu', 'lambda_j_mu_offset', 'lambda_j_sigma']
        self.rvs_b = ['beta']
        self.rvs_det = ['lambda_j']
        self.rvs_ppc = ['dhat_ij', 'shat_ij']

        return self.model


class CoxPH2(mt.BasePYMCModel):
    """Cox Proportional Hazards Piecewise Exponential model incl. censoring,
    CoxPH2 has a partial-pooled, strictly ordinal baseline hazard using a
    GaussianRandomWalk, and unpooled static covariates
    Core components used from oreum_core.model.BasePYMCModel
    As used by 301_Piecewise_CoxPH2.ipynb

    NOTE: We use a non-centered parameterisation because the data is relatively
    sparse, thus the FreeRV parmeters will be less well-informed, and we want
    to allow greater space of sampling. For more info see
    [Hierarchical Modeling, Betancourt, Nov 2020, Section 3.2]
    (https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html),
    a worked example from
    [Thomas Weicki's blog](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/),
    and more dicussion of the practical impacts on the
    [pymc Discourse](https://discourse.pymc.io/t/non-centering-results-in-lower-effective-sample-size/5850)

    """

    name = 'coxph2'
    version = '1.1.0'

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
        self.obs_nm = kwargs.pop('obs_nm', 'obs')
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
            (self.obs['duration'].max() // self.bin_width) + m + 1
        )

        # set immutable coords indep of dataset(a | b) i.e. ft & var names
        # start with factors
        self.coords = {k: list(d.keys()) for k, d in factor_map.items()}
        # add on the rest
        self.coords.update(
            dict(
                x_nm=self.obs.drop(self.fts_en + ['intercept'], axis=1).columns.values,
                j_nm=[
                    f'{v0:.0f}<t<={v1:.0f}'
                    for v0, v1 in zip(self.bin_edges, self.bin_edges[1:])
                ],
            )
        )

    def _build(self):
        """Build & return self.model, also set mutable coords i.e. obs_ids"""
        self.coords_m = dict(oid=self.obs.index.values)  # (44,)
        t_r = self.obs[self.fts_en[0]].values  # (44,)
        d_r = self.obs[self.fts_en[1]].values  # (44,)
        x_r = self.obs[self.coords['x_nm']].values  # (44, 1)

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
            t_ij = pm.MutableData('t_ij', t_ij_r, dims=('oid', 'j_nm'))  # (44, j)
            d_ij = pm.MutableData('d_ij', d_ij_r, dims=('oid', 'j_nm'))  # (44, j)
            x = pm.MutableData('x', x_r, dims=('oid', 'x_nm'))  # (44, 1)

            # 1. define baseline hazard as log-transformed hierarchical GaussianRandomWalk,
            # with non-centered parameterisation prior E(lj) ~ exp(-5) = 0.007
            l_j_m = pm.Normal('lambda_j_mu', mu=-5, sigma=1)
            l_j_m_o = pm.GaussianRandomWalk(
                'lambda_j_mu_offset',
                mu=0,
                sigma=1,
                dims='j_nm',
                init_dist=pm.Normal.dist(),
            )
            l_j_s = pm.InverseGamma('lambda_j_sigma', alpha=101, beta=100)
            l_j_ = pt.exp(l_j_m + (l_j_m_o * l_j_s))
            l_j = pm.Deterministic('lambda_j', l_j_, dims='j_nm')

            # 2. define covariate RVs
            b = pm.Normal('beta', mu=0, sigma=1, dims='x_nm')

            # 3. build up mu
            l_i_ = pt.exp(pt.dot(x, b.T))  # (44, )
            l_ij_ = pt.outer(l_i_, l_j)  # (44, j)
            mu_ = t_ij * l_ij_ + 1e-12  # avoid numerical errors near 0 (44, j)

            # 4. define likelihood
            _ = pm.Poisson('dhat_ij', mu=mu_, observed=d_ij, dims=('oid', 'j_nm'))

            # create predicted survival function for convenience
            _ = pm.Deterministic(
                'shat_ij', pt.exp(-pt.cumsum(mu_, axis=1)), dims=('oid', 'j_nm')
            )

        self.rvs_lam = ['lambda_j_mu', 'lambda_j_mu_offset', 'lambda_j_sigma']
        self.rvs_b = ['beta']
        self.rvs_det = ['lambda_j']
        self.rvs_ppc = ['dhat_ij', 'shat_ij']

        return self.model


class CoxPH3(mt.BasePYMCModel):
    """Cox Proportional Hazards Piecewise Exponential model incl. censoring,
    CoxPH2 has a partial-pooled, strictly ordinal baseline hazard lambda using a
    1D GaussianRandomWalk, and partial-pooled feature x coeffs beta that also use
    an xD GaussianRandomWalk (to generalise where x.dim > 1), but it's actually
    an MvGaussianRandomWalk with zero correlation
    Core components used from oreum_core.model.BasePYMCModel
    As used by 302_Piecewise_CoxPH2.ipynb

    NOTE: We use a non-centered parameterisation because the data is relatively
    sparse, thus the FreeRV parameters will be less well-informed, and we want
    to allow greater space of sampling. For more info see
    [Hierarchical Modeling, Betancourt, Nov 2020, Section 3.2]
    (https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html),
    a worked example from
    [Thomas Weicki's blog](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/),
    and more dicussion of the practical impacts on the
    [pymc Discourse](https://discourse.pymc.io/t/non-centering-results-in-lower-effective-sample-size/5850)

    """

    name = 'coxph3'
    version = '1.1.0'

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
        self.obs_nm = kwargs.pop('obs_nm', 'obs')
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
            (self.obs['duration'].max() // self.bin_width) + m + 1
        )

        # set immutable coords indep of dataset(a | b) i.e. ft & var names
        # start with factors
        self.coords = {k: list(d.keys()) for k, d in factor_map.items()}
        # add on the rest
        self.coords.update(
            dict(
                x_nm=self.obs.drop(self.fts_en + ['intercept'], axis=1).columns.values,
                j_nm=[
                    f'{v0:.0f}<t<={v1:.0f}'
                    for v0, v1 in zip(self.bin_edges, self.bin_edges[1:])
                ],
            )
        )

    def _build(self):
        """Build & return self.model, also set mutable coords i.e. obs_ids"""
        self.coords_m = dict(oid=self.obs.index.values)  # (i, )
        t_r = self.obs[self.fts_en[0]].values  # (i, )
        d_r = self.obs[self.fts_en[1]].values  # (i, )
        x_r = self.obs[self.coords['x_nm']].values  # (i, x)

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
            t_ij = pm.MutableData('t_ij', t_ij_r, dims=('oid', 'j_nm'))  # (i, j)
            d_ij = pm.MutableData('d_ij', d_ij_r, dims=('oid', 'j_nm'))  # (i, j)
            x = pm.MutableData('x', x_r, dims=('oid', 'x_nm'))  # (i, x)

            # 1. define baseline hazard as log-transformed hierarchical
            # GaussianRandomWalk with non-centered parameterisation
            l_j_m = pm.Normal('lambda_j_mu', mu=-5, sigma=1)  # ( ) 0d
            l_j_m_o = pm.GaussianRandomWalk(
                'lambda_j_mu_offset',
                mu=0,
                sigma=1,
                dims='j_nm',
                init_dist=pm.Normal.dist(),
            )  # (j, ) 1d
            l_j_s = pm.InverseGamma('lambda_j_sigma', alpha=101, beta=100)  # ( ) 0d
            l_j_ = pt.exp(l_j_m + (l_j_m_o * l_j_s))  # (j, ) 1d
            l_j = pm.Deterministic('lambda_j', l_j_, dims='j_nm')

            # 2. define covariate RVs
            b_m = pm.Normal('beta_mu', mu=-3, sigma=1, dims='x_nm')  # (x, )
            # used to need a hack to use MvGRW with unpooled var to create
            # unpooled GRWs. In pymc 5.16.* this is no longer needed, but keep
            # the code around for future reference in case useful to have
            # correlated MvGRWs
            # bt_s = pm.InverseGamma("bt_s", alpha=101, beta=10, dims="xt_names")
            # bt = pm.MvGaussianRandomWalk(
            #     "bt", mu=0, chol=tt.diag(bt_s), dims=["t_name", "xt_names"]
            # )
            b_m_o = pm.GaussianRandomWalk(
                'beta_mu_offset',
                mu=0,
                sigma=1,
                dims=('x_nm', 'j_nm'),
                init_dist=pm.Normal.dist(),
            )
            b_s = pm.InverseGamma('beta_sigma', alpha=101, beta=100, dims='x_nm')
            b_ = (b_m + (b_m_o.T * b_s)).T  # (x, j)
            b = pm.Deterministic('beta', b_, dims=('x_nm', 'j_nm'))

            # 3. build up mu
            l_i_ = pt.exp(pt.dot(x, b))  # (i, x) . (x, j) -> (i, j)
            l_ij_ = l_i_ * l_j  # (i, j) * (j, ) -> (i, j)
            mu_ = t_ij * l_ij_ + 1e-9  # (i, j)

            # 4. define likelihood
            _ = pm.Poisson('dhat_ij', mu=mu_, observed=d_ij, dims=('oid', 'j_nm'))

            # # create predicted survival function for convenience
            _ = pm.Deterministic(
                'shat_ij', pt.exp(-pt.cumsum(mu_, axis=1)), dims=('oid', 'j_nm')
            )

        self.rvs_lam = ['lambda_j_mu', 'lambda_j_mu_offset', 'lambda_j_sigma']
        self.rvs_b = ['beta_mu', 'beta_mu_offset', 'beta_sigma']
        self.rvs_det = ['lambda_j', 'beta']
        self.rvs_ppc = ['dhat_ij', 'shat_ij']

        return self.model
