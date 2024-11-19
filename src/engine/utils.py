# src.engine.utils.py
# copyright 2024 Oreum OÜ
"""Assorted display and plotting utilities for the project"""
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from matplotlib import figure
from oreum_core import curate
from oreum_core import model_pymc as mt

__all__ = [
    'display_rvs',
    'get_fs',
    'plot_f',
    'ProjectDFXCreator',
    'get_synthetic_df_oos',
    'get_ppc_components_as_dfm',
]

sns.set_theme(
    style='darkgrid',
    palette='muted',
    context='notebook',
    rc={'figure.dpi': 72, 'savefig.dpi': 144, 'figure.figsize': (12, 4)},
)


def display_rvs(mdl: mt.BasePYMCModel):
    """Convenience to display RVS and values"""
    _ = display(f'RVs for {mdl.mdl_id}')
    if mdl.model is not None:
        _ = [display(Markdown(s)) for s in mt.print_rvs(mdl)]


def get_fs(fd, param: dict[str, float]) -> pd.DataFrame:
    """Convenience to create SF and h functions for fixed distribution fd
    between ppf quantile lims"""
    t = np.linspace(fd.ppf(0.03), fd.ppf(0.97), 95)
    param.update({'t': t, 'event_density_pi': fd.pdf(t), 'survival_s': fd.sf(t)})
    dfs = pd.DataFrame(param, index=np.arange(95))
    dfs['hazard_lambda'] = dfs['event_density_pi'] / dfs['survival_s']
    dfs['csum_hazard_lambda'] = dfs['hazard_lambda'].cumsum()
    return dfs


def plot_f(dff: pd.DataFrame, f_nm: str, p_nm: str, **kwargs) -> figure.Figure:
    """Convenience to plot functions created by get_fs"""
    dfm = dff.melt(id_vars=[p_nm, 't'], var_name='f', value_name='density')
    lvls = ['hazard_lambda', 'csum_hazard_lambda', 'survival_s', 'event_density_pi']
    dfm['f'] = pd.Categorical(dfm['f'], categories=lvls, ordered=True)
    pal = kwargs.get('palette', 'magma_r')

    kws = dict(
        palette=pal,
        lw=5,
        alpha=0.9,
        kind='line',
        col_wrap=2,
        legend='full',
        height=3,
        aspect=2,
        facet_kws=dict(sharex=True, sharey=False),
    )
    g = sns.relplot(x='t', y='density', hue=p_nm, col='f', data=dfm, **kws)
    _ = g.fig.suptitle(
        f'{f_nm} for a range of ${p_nm} \in (0, \infty)$ over qs [0.01, 0.99]'
    )
    _ = g.tight_layout()
    return g.fig


class ProjectDFXCreator:
    """Convenience class to process the main dataset (mastectomy) that's used
    throughout this project into `dfx` for model input. Saves us having to
    redeclare it in each notebook.
    """

    def __init__(self):
        # NOTE: The death event is an endog feature, but is recorded in our data
        # as a bool so for convenience we will patsy transform the death event
        # as if it were part of the linear model, then remove it from fts_ex and
        # put into fts_en, and rearrange dfx accordingly also
        self.fml_tfmr = '1 + met + death'
        self.fts_en0 = ['duration']
        self.fts_en = ['duration', 'death_t_true']
        self.dfcmb = None
        self.dfscale = None
        self.factor_map = None

    def get_dfx(self, df: pd.DataFrame, in_sample: bool = True) -> pd.DataFrame:
        """Reshape, transform & standardize df to create dfx for model input"""

        # 1. Create dfcmb (reshaped) from df
        if in_sample:
            reshaper = curate.DatasetReshaper()
            self.dfcmb = reshaper.create_dfcmb(df)
        else:
            assert self.dfcmb is not None, "run in_sample = True first"

        # 2. create Transformer based on dfcmb
        tfmr = curate.Transformer()
        _ = tfmr.fit_transform(self.fml_tfmr, self.dfcmb)
        self.factor_map = tfmr.factor_map

        # 3. Transform df according to dfcmb
        df_ex = tfmr.transform(df, propagate_nans=True)

        # 5. Standardize
        stdr = curate.Standardizer(tfmr)
        if in_sample:
            df_exs = stdr.fit_standardize(df_ex)
            self.dfscale, _ = stdr.get_scale()  # unused in this process, because no dfb
        else:
            stdr.set_scale(self.dfscale)
            df_exs = stdr.standardize(df_ex)

        df_en = df[self.fts_en0].copy()
        kws_mrg = dict(how='inner', left_index=True, right_index=True)
        dfx = pd.merge(df_en, df_exs, **kws_mrg).rename(
            columns=lambda x: tfmr.snl.clean_patsy(x)
        )

        fts_ex = [v for v in dfx.columns.values if v not in self.fts_en]
        dfx = dfx.reindex(columns=self.fts_en + fts_ex)

        return dfx


def get_synthetic_df_oos(
    df: pd.DataFrame, is_coxph: bool = False, ndur: int = 10
) -> pd.DataFrame:
    """Convenience to create synthetic df for out-of-sample
    sample-ppc forecast work. This function knows too much about the
    dataset, but it's only a convenience for this little project"""
    tmax = df['duration'].max()
    dfo = pd.DataFrame(
        {
            'met': [False, True],
            'death': [False, False],
            'pid': [f'z{str(i).zfill(3)}' for i in range(2)],
        }
    )
    if is_coxph:  # simple dataset, explode not needed
        dfo['duration'] = tmax

    else:  # will need to explode durations
        l_dur = np.round(np.linspace(0, tmax + 1, ndur), 0).tolist()
        dfo['duration'] = [l_dur for i in range(len(dfo))]
        dfo = dfo.explode('duration')
        dfo['duration'] = dfo['duration'].astype(int)
        dfo['pid'] = dfo['pid'].astype(str) + '-' + dfo['duration'].astype(str)

    dfo.set_index('pid', inplace=True)
    return dfo


def get_ppc_components_as_dfm(
    idata: az.InferenceData,
    getx: bool = False,
    is_coxph: bool = False,
    j_nm: list = None,
) -> pd.DataFrame:
    """Convenience to extract rvs_ppc from idata.posterior_predictive for
    subsequent analysis and plotting. This function knows too much about the
    model architecture internals and naming, but it's only a convenience for
    this little project
    """
    rvs_ppc = ['shat_ij', 'dhat_ij'] if is_coxph else ['shat', 'that']
    dfhat = (
        az.extract(idata, group='posterior_predictive', var_names=rvs_ppc)
        .to_dataframe()[rvs_ppc]
        .reset_index()
    )
    dfhat['oid_sub'] = dfhat['oid'].apply(lambda x: x.split('-')[0])

    rv_t = 't_ij' if is_coxph else 't'
    df_t = idata.constant_data[rv_t].to_dataframe().reset_index()

    if is_coxph:
        df_t['t'] = df_t.groupby('oid')['t_ij'].cumsum()
        idx = df_t['t_ij'] == 0
        df_t.loc[idx, 't'] = 0
        df_t['tau_min'] = df_t['j_nm'].apply(lambda x: int(x.split('<t<=')[0]))
        df_t['tau_max'] = df_t['j_nm'].apply(lambda x: int(x.split('<t<=')[1]))
        dfout = pd.merge(dfhat, df_t, how='left', on=['oid', 'j_nm'])
        dfout['j_nm'] = pd.Categorical(dfout['j_nm'], categories=j_nm, ordered=True)
        dfout.sort_values(['oid', 'j_nm'], inplace=True)

    else:
        dfout = pd.merge(dfhat, df_t, how='left', on='oid')

    if getx:
        df_x = (
            idata.constant_data['x'].to_dataframe().unstack('x_nm').droplevel(0, axis=1)
        )
        df_x.columns.name = None
        df_x.reset_index(inplace=True)
        dfout = pd.merge(dfout, df_x, how='left', on='oid')

    return dfout
