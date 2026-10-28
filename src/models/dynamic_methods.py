import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from dataclasses import dataclass
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
class DynamicFMResults:
    """
    Container for the output of DynamicFM.

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

def DynamicFM(data: pd.DataFrame, nfactors: int, factor_orders: int,
              idiosyncratic_ar1: bool | None = False,
              missing: str | None = 'fill-em', maxiter: int | None = 500,
              tolerance: float | None = 1e-6, **kwargs) -> DynamicFMResults:
    """
    Estimate a dynamic factor model via the EM algorithm of statsmodels' DynamicFactorMQ,
    initialized with the static PCA estimates.

    The pipeline is:
        1. Standardize the data.
        2. Estimate the factors and the loadings via PCA (SFM_StateSpace).
        3. Estimate the VAR process of the factors of the given order.
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
    factor_orders : int
        Order of the VAR process of the factors.
    idiosyncratic_ar1 : bool | None, optional
        Whether the idiosyncratic components follow an AR(1) process. The default is False.
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
    DynamicFMResults
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
    factor_orders = int_like(factor_orders, 'factor_orders')
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

    return DynamicFMResults(results=dfm_res, start_params=init_params, var_order=factor_orders,
                            mu=mu, sigma=sigma)
