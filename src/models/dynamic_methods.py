import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from dataclasses import dataclass
from scipy.stats import chi2
from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          bool_like,
                                          float_like,
                                          int_like,
                                          )
from src.models.static_methods import SFM_StateSpace

## Global attributes
plt.rc('figure', figsize=(16,6))
plt.rc('lines', linewidth=1.5)
sns.set_style('darkgrid')

@dataclass
class DFM_StateSpace_Results:
    """
    Container for the output of DFM_StateSpace.

    Attributes
    ----------
    results : object
        Fitted statsmodels DynamicFactorMQResults instance. Note that it is fitted on the
        standardized data, hence its forecasts need to be rescaled via mu and sigma.
    start_params : np.ndarray
        Initial parameters fed to the EM algorithm.
    var_order : int
        Order of the VAR process followed by the common factors.
    mu : pd.Series
        Sample means of the series used to standardize the data.
    sigma : pd.Series
        Sample standard deviations of the series used to standardize the data.
    """
    results: object
    start_params: np.ndarray
    var_order: int
    mu: pd.Series
    sigma: pd.Series

def select_var_order(factors: pd.DataFrame, maxlags: int | None = 8,
                     alpha: float | None = 0.05) -> int:
    """
    Select the order of the VAR process followed by the common factors.

    The order is first chosen via AIC. It is then compared against the next most
    parsimonious process (the AIC selected order minus one) via the likelihood ratio test.
    The larger process is kept only if the null of the smaller one is rejected at level alpha.

    Parameters
    ----------
    factors : pd.DataFrame
        Estimated common factors, one column per factor.
    maxlags : int | None, optional
        Upper bound for the order of the process. The default is 8.
    alpha : float | None, optional
        Significance level of the likelihood ratio test. The default is 0.05.

    Returns
    -------
    int
        Selected VAR order. Always at least 1, as required by DynamicFactorMQ.

    """
    if type(factors) != pd.core.frame.DataFrame:
        raise ValueError(f'The factors must be a pandas DataFrame. Got {type(factors)}.')
    maxlags = int_like(maxlags, 'maxlags')
    alpha = float_like(alpha, 'alpha')

    freq = getattr(factors.index, 'freq', None)
    state_var = sm.tsa.VAR(factors, freq=freq)
    lag_aic = state_var.select_order(maxlags=maxlags, trend='n').aic

    ## A VAR(1) cannot be reduced any further. A VAR(0) is not admissible for the state equation
    if lag_aic <= 1:
        return 1

    ## The smaller model is fit on one less observation so that both are estimated on the same sample
    fit1 = sm.tsa.VAR(factors, freq=freq).fit(maxlags=lag_aic, ic=None, trend='n')
    fit2 = sm.tsa.VAR(factors.iloc[1:,:], freq=freq).fit(maxlags=lag_aic-1, ic=None, trend='n')
    lr_stat = -2 * (fit2.llf - fit1.llf)
    chi_df = factors.shape[1] ** 2
    lr_pval = chi2.sf(lr_stat, chi_df)

    return int(lag_aic) if lr_pval < alpha else int(lag_aic) - 1

def DFM_init_params(loadings: pd.DataFrame, var_params: pd.DataFrame,
                    resid_var: pd.Series, idio_ar1: pd.Series | None = None) -> np.ndarray:
    """
    Assemble the vector of starting parameters for DynamicFactorMQ out of the static estimates.
    Even though statsmodels by default uses PCA to initialize the EM algorithm, I still
    prefer to feed the initial parameters myself.

    Parameters
    ----------
    loadings : pd.DataFrame
        Matrix of loadings of shape (nvar, nfactors).
    var_params : pd.DataFrame
        Coefficients of the VAR process of the factors as returned by the params attribute
        of the statsmodels VAR results, i.e. of shape (nfactors*order, nfactors).
    resid_var : pd.Series
        Variances of the idiosyncratic components. If idio_ar1 is provided these should be the
        unconditional variances, which are converted here to the variances of the innovations.
    idio_ar1 : pd.Series | None, optional
        AR(1) coefficients of the idiosyncratic components. Only to be provided if the model
        is specified with idiosyncratic_ar1=True. The default is None.

    Returns
    -------
    np.ndarray
        Parameters in the order expected by DynamicFactorMQ with a single block of global factors.

    """
    loadings = array_like(loadings, 'loadings', ndim=2)
    var_params = array_like(var_params, 'var_params', ndim=2)
    resid_var = array_like(resid_var, 'resid_var', ndim=1)
    idio_ar1 = array_like(idio_ar1, 'idio_ar1', ndim=1, optional=True)
    nfactors = loadings.shape[1]

    '''
    The parameters are passed in the following order, all vectorized:

        1. Matrix of loadings, variable by variable, vectorized with loadings.ravel()
        2. Transition matrices of the VAR process, equation by equation with the lags side by side [A_1 A_2 ... A_p]
           Since statsmodels stacks the lags vertically as [A_1.T; A_2.T; ...; A_p.T], this is just params.T.ravel()
        3. Lower triangular entries of the Cholesky factor of the covariance of the factor innovations
           *Under the identification scheme F.T @ F / T = I this is the identity matrix
        4. Coefficients of the AR(1) process of the idiosyncratic components - if idiosyncratic_ar1 is True
        5. Variances of the idiosyncratic components (of the innovations if idiosyncratic_ar1 is True)
    '''
    init_params = loadings.ravel()
    init_params = np.append(init_params, var_params.T.ravel())
    init_params = np.append(init_params, np.identity(nfactors)[np.tril_indices(nfactors, 0)])

    if idio_ar1 is not None:
        init_params = np.append(init_params, idio_ar1)
        resid_var = resid_var * (1 - idio_ar1 ** 2)
    init_params = np.append(init_params, resid_var)

    return init_params

def DFM_StateSpace(data: pd.DataFrame, nfactors: int, factor_orders: int | None = None,
                   idiosyncratic_ar1: bool | None = False, maxlags: int | None = 8,
                   missing: str | None = 'fill-em', maxiter: int | None = 500,
                   tolerance: float | None = 1e-6, **kwargs) -> DFM_StateSpace_Results:
    """
    Estimate a dynamic factor model via the EM algorithm of statsmodels' DynamicFactorMQ,
    initialized with the static PCA estimates.

    The pipeline is:
        1. Standardize the data.
        2. Estimate the factors and the loadings via PCA (SFM_StateSpace).
        3. Select the order of the VAR process of the factors (select_var_order) and estimate it.
        4. Estimate the variances (and AR(1) coefficients) of the idiosyncratic components.
        5. Fit DynamicFactorMQ starting from the above estimates (DFM_init_params).

    Reference: Bańbura & Modugno (2014), Maximum likelihood estimation of factor models
    on datasets with arbitrary pattern of missing data.

    Parameters
    ----------
    data : pd.DataFrame
        Stationary data indexed by time with a recognizable frequency. Missing values are allowed,
        they are only imputed for the initialization and are handled natively by the Kalman filter.
    nfactors : int
        Number of common factors, e.g. as estimated via ABC_crit.
    factor_orders : int | None, optional
        Order of the VAR process of the factors. If None it is selected via select_var_order.
        The default is None.
    idiosyncratic_ar1 : bool | None, optional
        Whether the idiosyncratic components follow an AR(1) process. The default is False.
    maxlags : int | None, optional
        Upper bound for the order selection. The default is 8.
    missing : str | None, optional
        Imputation method for the initialization, 'mean' or 'fill-em'. The default is 'fill-em'.
    maxiter : int | None, optional
        Maximum number of EM iterations. The default is 500.
    tolerance : float | None, optional
        Convergence threshold of the EM algorithm. The default is 1e-6.
    **kwargs
        Passed on to the constructor of DynamicFactorMQ.

    Returns
    -------
    DFM_StateSpace_Results
        Dataclass containing the fitted results, the starting parameters, the VAR order
        and the moments used to standardize the data.

    Raises
    ------
    ValueError
        If data is not a pandas DataFrame.

    """
    if type(data) != pd.core.frame.DataFrame:
        raise ValueError(f'The data must be a pandas DataFrame. Got {type(data)}.')
    nfactors = int_like(nfactors, 'nfactors')
    factor_orders = int_like(factor_orders, 'factor_orders', optional=True)
    idiosyncratic_ar1 = bool_like(idiosyncratic_ar1, 'idiosyncratic_ar1')
    missing = string_like(missing, 'missing')
    maxiter = int_like(maxiter, 'maxiter')
    tolerance = float_like(tolerance, 'tolerance')

    ## Standardize the data. The statsmodels standardization is turned off so that the moments used are known
    mu = data.mean()
    sigma = data.std(ddof=0)
    data_std = (data - mu) / sigma

    ## Rows with fewer than nfactors observations (e.g. COVID quarters set to NaN) make the static EM imputation infeasible.
    ## They are left out of the initialization only, the Kalman filter of DynamicFactorMQ handles them natively.
    keep = (data_std.notna().sum(axis=1) >= nfactors).to_numpy()
    data_init = data_std.loc[keep]
    if not keep.all():
        data_init = data_init.reset_index(drop=True)

    ## Static estimates under the identification scheme (F.T @ F)/T = I with the first row of the loadings positive
    sfm = SFM_StateSpace(data_init.copy(), nfactors=nfactors, demean=False, standardize=False,
                         missing=missing, normalize=True)
    factors, loadings = sfm.factors, sfm.loadings

    ## VAR process of the factors
    if factor_orders is None:
        factor_orders = select_var_order(factors, maxlags=maxlags)
    var_res = sm.tsa.VAR(factors, freq=getattr(factors.index, 'freq', None)).fit(maxlags=factor_orders, ic=None, trend='n')

    ## Idiosyncratic components. Only the residuals at the observed entries are used
    resid = data_init - np.asarray(factors) @ np.asarray(loadings).T
    resid_var = resid.var(ddof=0)
    idio_ar1 = None
    if idiosyncratic_ar1:
        idio_ar1 = pd.Series({c: resid[c].autocorr(lag=1) for c in resid.columns})

    init_params = DFM_init_params(loadings, var_res.params, resid_var, idio_ar1=idio_ar1)

    dfm_model = sm.tsa.DynamicFactorMQ(data_std, factor_multiplicities=nfactors,
                                       factor_orders=factor_orders, standardize=False,
                                       idiosyncratic_ar1=idiosyncratic_ar1, **kwargs)
    dfm_res = dfm_model.fit(start_params=init_params, transformed=True, cov_type='none',
                            method='em', maxiter=maxiter, tolerance=tolerance)

    return DFM_StateSpace_Results(results=dfm_res, start_params=init_params, var_order=factor_orders,
                            mu=mu, sigma=sigma)
