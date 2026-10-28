# unibo-dfm-x-py
A Python framework for macroeconomic forecasting and nowcasting with dynamic factor models (DFMs).

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Workflow](#workflow)
- [Usage Example](#usage-example)
- [Models](#models)
- [Extra Modules](#extra-modules)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

## Overview
This repository develops a framework in Python for estimating dynamic factor models on large macroeconomic panels, such as the FRED-QD and FRED-MD databases, and for comparing their out-of-sample forecasts through a rolling-window scheme. The factors and the loadings are initialized via principal components, and the state space model is estimated via the Kalman filter and the EM algorithm of the [`DynamicFactorMQ`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ.html) class from `statsmodels`.

## Installation
```bash
git clone https://github.com/sr-majidi/unibo-dfm-x-py.git
cd unibo-dfm-x-py
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install numpy pandas scipy matplotlib seaborn statsmodels
```

See [Requirements](#requirements) for the list of required packages. The scripts have to be run from the root of the repository, as the imports are relative to it.

## Workflow
| Step | Stage | Code | Produces |
|---|---|---|---|
| 1 | Prepare the data | [`Fred_transform`](#fred_transform) on the raw FRED csv, followed by the cleaning of the missing and non-stationary series (see `main.py`) | a stationary dataframe indexed by time |
| 2 | Estimate the number of common factors | [`ABC_crit`](#abc_crit) | `rhat1` and `rhat2`, the number of factors at the 5% and 1% thresholds |
| 3 | Static factor analysis | [`SFM_StateSpace`](#class-sfm_statespace) / [`StaticFM`](#staticfm) | factors, loadings, projection, residuals, eigenvalues and eigenvectors |
| 4 | VAR lag selection | [`select_var_order`](#select_var_order) | the order of the VAR process of the factors, chosen by AIC and confirmed via the likelihood ratio test |
| 5 | State space estimation | [`DFM_StateSpace`](#dfm_statespace), which calls [`DFM_init_params`](#dfm_init_params) and `DynamicFactorMQ` | a [`DFM_StateSpace_Results`](#class-dfm_statespace_results) instance holding the fitted model |
| 6 | Rolling-window forecasting | `results.forecast(...)` on every window, rescaled via `mu` and `sigma` | 1-step and 4-step ahead forecasts for every forecast origin, the mean squared forecast errors of the select variables in a pandas dataframe, and plots of the forecasts against the real observations |
| 7 | Spectral analysis and unit root diagnostics (optional) | [`DFM_periodogram`](#dfm_periodogram) on the estimated factors, [`adf_wrapper`](#unit-root-diagnostics), [`kpss_wrapper`](#unit-root-diagnostics), [`CT_unitroot`](#unit-root-diagnostics) | periodograms of the factors, ADF/KPSS test results and the wild bootstrap ADF p-value |

## Usage Example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.modules.Fred_transform import Fred_transform
from src.modules.ABC_crit import ABC_crit
from src.models.dynamic_methods import DFM_StateSpace, select_var_order
from src.models.static_methods import SFM_StateSpace

## Load the raw FRED-QD data and apply the stationarity transformations
df = pd.read_csv('./data/2026-04-QD.csv')
df = df.iloc[:-1,:] ## Often for some variables the observation corresponding to the last observation date is not yet recorded
fred_df = Fred_transform(df, freq='QD')
fred_df.index.freq = fred_df.index.inferred_freq

## The first two observations are lost due to the I(2) transformations
fred_df = fred_df.iloc[2:,:]
```
> [!NOTE]
> The data used in `main.py` further goes through the removal of the series with missing values and of the series that are not stationary even after the transformations. See `main.py` for the full procedure. The remainder of this example assumes that `fred_df` is the cleaned dataframe.

```python
## Estimate the number of common factors via the ABC criterion
T, n = fred_df.shape
kf = int((min(n,T)/100)**0.25) ## Bai and Ng (2002) upper bound for kmax
kmax = 8 * kf if kf > 0 else 8

fig, ax = plt.subplots()
rhat1, rhat2, ax = ABC_crit(fred_df, kmax, seed=1776, ax=ax, standardize=True)
plt.show()

kfactors = rhat1

## Static factor analysis. The data is standardized internally
sfm = SFM_StateSpace(fred_df, nfactors=kfactors, standardize=True, method='svd', missing='fill-em', normalize=True)
sfm.plot_scree()
plt.show()

## Select the order of the VAR process followed by the common factors
state_lag = select_var_order(sfm.factors, maxlags=4, alpha=0.05)

## Estimate the dynamic factor model, initialized with the static estimates
dfm = DFM_StateSpace(fred_df, nfactors=kfactors, factor_orders=state_lag, idiosyncratic_ar1=False)
dfm_res = dfm.results
print(dfm_res.summary())
```
> [!NOTE]
> 1. The data is standardized by `DFM_StateSpace` before the model is estimated, therefore the forecasts are on the standardized scale and have to be brought back to the original scale via the moments stored in the output.
>
> 2. The upper bound for the order of the VAR process is set to 4, which is one year of quarterly lags. With a larger bound the AIC tends to select the largest order available, and the EM algorithm can fail on some of the estimation windows.

```python
## 4-step ahead forecasts, rescaled to the original scale of the data
fc = dfm_res.forecast(steps=4)
fc = fc * dfm.sigma + dfm.mu

## The factors estimated via the Kalman smoother can be accessed via
factors_smoothed = dfm_res.factors.smoothed

## Spectrum of the estimated common factors, one panel per factor
from src.modules.DFM_periodogram import DFM_periodogram

fig, ax = plt.subplots(nrows=kfactors, sharex=True, figsize=(16, 4*kfactors)) ## The figure is made taller to fit the labels of the stacked plots
for k in range(kfactors):
    DFM_periodogram(sfm.factors.iloc[:,k], freq='QD', ax=ax[k])
plt.show()
```

The rolling-window forecasts are obtained by re-estimating the model on the most recent `W` observations at each forecast origin. Continuing the example, with `fred_df_full` denoting the cleaned dataframe before it is truncated to the first window:
```python
W = 80 ## Window size, 20 years of quarterly data
horizons = [1, 4]
H = max(horizons)
series_names = ['GDPC1', 'PCECC96', 'UNRATE', 'DPIC96', 'INDPRO', 'HOUST', 'TB3MS'] ## Select variables

fc_dfm = {h: [] for h in horizons}
fc_index = {h: [] for h in horizons}

for origin in range(W, fred_df_full.shape[0] - H + 1):
    window = fred_df_full.iloc[origin-W:origin,:]
    window.index.freq = window.index.inferred_freq
    dfm_w = DFM_StateSpace(window, nfactors=kfactors, factor_orders=state_lag, maxiter=50)
    fc = dfm_w.results.forecast(steps=H) * dfm_w.sigma + dfm_w.mu
    for h in horizons:
        fc_dfm[h].append(fc.iloc[h-1,:].to_numpy())
        fc_index[h].append(fred_df_full.index[origin+h-1])

## Forecasts of each horizon indexed by the date of the target observation
fc_df = {h: pd.DataFrame(np.array(fc_dfm[h]), index=fc_index[h], columns=fred_df_full.columns) for h in horizons}

## Mean squared forecast errors of the select variables, one column per forecast horizon
msfe = pd.DataFrame({f'h = {h}': ((fred_df_full.loc[fc_df[h].index, series_names] - fc_df[h][series_names])**2).mean()
                     for h in horizons})
print(msfe)
```
> [!NOTE]
> 1. The EM algorithm is capped at 50 iterations (`maxiter=50`) when the model is re-estimated at each forecast origin, as a full run of 500 iterations on every window is prohibitively slow.
>
> 2. The COVID quarters are set to NaN in the observations and are skipped in the averages of the squared forecast errors. They show up as gaps in the plots of the actual series. The mean squared forecast errors are on the scale of the transformed data and are therefore not comparable across variables.

## Models

### `class SFM_StateSpace`
```python
class SFM_StateSpace(data: pd.DataFrame,
                     nfactors: int | None = None,
                     demean: bool | None = False,
                     standardize: bool | None = True,
                     method: str | None = 'svd',
                     missing: str | None = 'fill-em',
                     normalize: bool | None = True,
                     loadings_mat: object | None = None,
                     tol_em: float | None = 5e-8,
                     max_em_iter: int | None = 1000)
```
Static factor model class. It handles the missing values, the standardization of the data, the principal components analysis and the identification of the factors and the loadings.
#### Parameters
- `data`: Data matrix of size T x n. 
- `nfactors`: Number of common factors. If not provided, it is set to the upper bound suggested by Bai & Ng (2002).
- `demean`: Whether to demean the data. The default is False.
- `standardize`: Whether to standardize the data. The default is True.
- `method`: Decomposition method, either `'svd'` for the singular value decomposition of the data matrix or `'eig'` for the eigendecomposition of the sample covariance matrix. 
- `missing`: Method to handle the missing values, either `'mean'` or `'fill-em'`. The latter is an iterative EM algorithm based on PCA.
- `normalize`: If True, the identification scheme is such that the factors have unit variance, i.e. `(F.T @ F)/T = I`, and the loadings of the first variable are all positive. Otherwise, the inner product of the matrix of loadings is the identity.
- `loadings_mat`: Optional. The true matrix of loadings, if known, to compare the estimated loadings against it.
- `tol_em`: Convergence threshold of the EM algorithm used for imputing the missing values.
- `max_em_iter`: Maximum number of iterations of the EM algorithm used for imputing the missing values.

#### Methods
- `project(nfactors, transform)`: Projection of the data on the span of the factors, optionally brought back to the original scale of the data.
- `plot_scree(ncomp)`: Scree plot of the singular values or the eigenvalues, scaled by the square root of T.
- `loadings_subspace_angles()`: Subspace angles between the estimated and the original matrix of loadings. Requires `loadings_mat` to be provided.

#### Properties
- `factors`: Estimated common factors.
- `loadings`: Estimated matrix of loadings.
- `projection`: The estimated common component, in the original units of the data.
- `residuals`: The estimated idiosyncratic component, in the original units of the data.
- `standardized_residuals`: The estimated idiosyncratic component on the transformed scale.
- `transformed_data`: The data after demeaning or standardization.
- `eigenvals`: Eigenvalues of the sample covariance matrix (squared singular values of the data matrix).
- `eigenvecs`: The corresponding eigenvectors.

### `StaticFM`
Wrapper function around the `SFM_StateSpace` class. It takes the same arguments as the class, apart from the ones controlling the EM algorithm, and returns the tuple `(factors, loadings, projection, residuals, eigenvals, eigenvecs)`.
```python
from src.models.static_methods import StaticFM
factors, loadings, projection, residuals, eigenvals, eigenvecs = StaticFM(fred_df, nfactors=kfactors)
```

### `select_var_order`
```python
select_var_order(factors: pd.DataFrame,
                 maxlags: int | None = 8,
                 alpha: float | None = 0.05) -> int
```
This function selects the order of the VAR process followed by the common factors. The order is first chosen via AIC. It is then compared against the next most parsimonious process, i.e. the AIC selected order minus one, via the likelihood ratio test. The larger process is kept only if the null hypothesis of the smaller one is rejected at level `alpha`.

Parameters:
- `factors`: Dataframe of the estimated common factors, one column per factor.
- `maxlags`: Upper bound for the order of the process. The default is 8.
- `alpha`: Significance level of the likelihood ratio test. The default is 0.05.

Returns:
- `int` containing the selected VAR order. It is always at least 1, as required by `DynamicFactorMQ`.

### `DFM_init_params`
```python
DFM_init_params(loadings: pd.DataFrame,
                var_params: pd.DataFrame,
                resid_var: pd.Series,
                idio_ar1: pd.Series | None = None) -> np.ndarray
```
This function assembles the vector of starting parameters for `DynamicFactorMQ` out of the static estimates. The parameters have to be passed in the following order, with all the matrices vectorized:
1. The matrix of loadings, variable by variable.
2. The transition matrices of the VAR process, equation by equation with the lags side by side.
3. The lower triangular entries of the Cholesky factor of the covariance of the factor innovations. Under the identification scheme `(F.T @ F)/T = I` this is the identity matrix.
4. The coefficients of the AR(1) process of the idiosyncratic components, only if `idiosyncratic_ar1` is True.
5. The variances of the idiosyncratic components, or those of their innovations if `idiosyncratic_ar1` is True.

Parameters:
- `loadings`: Matrix of loadings of shape (nvar, nfactors).
- `var_params`: Coefficients of the VAR process as returned by the `params` attribute of the `statsmodels` VAR results.
- `resid_var`: Variances of the idiosyncratic components.
- `idio_ar1`: Optional. AR(1) coefficients of the idiosyncratic components.

> [!NOTE]
> `statsmodels` orders the loadings variable by variable, therefore the matrix of loadings has to be flattened as `loadings.ravel()` and not as `loadings.T.ravel()`.

### `DFM_StateSpace`
```python
DFM_StateSpace(data: pd.DataFrame,
               nfactors: int,
               factor_orders: int | None = None,
               idiosyncratic_ar1: bool | None = False,
               maxlags: int | None = 8,
               missing: str | None = 'fill-em',
               maxiter: int | None = 500,
               tolerance: float | None = 1e-6,
               **kwargs) -> DFM_StateSpace_Results
```
Wrapper function around the `DynamicFactorMQ` class of `statsmodels`. It standardizes the data, estimates the factors and the loadings via PCA, selects and estimates the VAR process of the factors, estimates the variances of the idiosyncratic components, and finally fits the model via the EM algorithm starting from the static estimates.

Parameters:
- `data`: Stationary dataframe indexed by time with a recognizable frequency. Missing values are allowed, they are only imputed for the initialization and are handled natively by the Kalman filter.
- `nfactors`: Number of common factors, e.g. as estimated via `ABC_crit`.
- `factor_orders`: Order of the VAR process of the factors. If not provided, it is selected via `select_var_order`.
- `idiosyncratic_ar1`: Whether the idiosyncratic components follow an AR(1) process. The default is False.
- `maxlags`: Upper bound for the order selection. The default is 8.
- `missing`: Imputation method for the initialization, `'mean'` or `'fill-em'`. The default is `'fill-em'`.
- `maxiter`: Maximum number of EM iterations. The default is 500.
- `tolerance`: Convergence threshold of the EM algorithm. The default is 1e-6.
- `**kwargs`: Passed on to the constructor of `DynamicFactorMQ`.

Returns:
- `DFM_StateSpace_Results` containing the fitted results, the starting parameters, the VAR order and the moments used to standardize the data.

> [!NOTE]
> The rows of the data with fewer than `nfactors` observations, e.g. the COVID quarters set to NaN, are left out of the static initialization only. The Kalman filter of `DynamicFactorMQ` uses all the rows.

### `class DFM_StateSpace_Results`
Results container for the output of `DFM_StateSpace`.
#### Properties
- `results`: Fitted `DynamicFactorMQResults` instance from `statsmodels`. It is fitted on the standardized data.
- `start_params`: Initial parameters fed to the EM algorithm.
- `var_order`: Order of the VAR process followed by the common factors.
- `mu`: Sample means of the series used to standardize the data.
- `sigma`: Sample standard deviations of the series used to standardize the data.

### `transition_mat`
```python
transition_mat(dimension: int,
               order: int,
               stable: bool | None = False,
               alpha: float | None = None,
               seed: int | None = 1776)
```
This function generates random transition matrices in the companion form for simulating VAR processes. If `stable` is True, the companion matrix is rescaled by its spectral radius to ensure stability. It is used for testing and validating the dynamic factor model machinery on simulated data.

## Extra Modules

### `Fred_transform`
This function transforms each of the series in the FRED QD and MD datasets according to their transformation codes and returns a clean pandas dataframe containing the transformed series indexed by time.
- The function takes as input the raw dataset downloaded from the FRED database and loaded via pandas.
- Additionally, the function takes a two character string input ('QD' or 'MD') to determine the frequency of the input data.

```python
import pandas as pd
from src.modules.Fred_transform import Fred_transform

## Quarterly data
fred_qd = pd.read_csv('./data/2026-04-QD.csv')
fred_qd_tr = Fred_transform(fred_qd, 'QD')

## Monthly data
fred_md = pd.read_csv('./data/2026-04-MD.csv')
fred_md_tr = Fred_transform(fred_md, 'MD')
```

### `ABC_crit`
This function implements the information criterion of Alessi, Barigozzi and Capasso (2010).
- The input data has to be a pandas DataFrame object. It should not contain missing values or else the function throws out an error.
- The upper bound for the number of factors `kmax` is a required input and must provided.
- If `ax` is provided, the function also plots the IC on the same canvas on `ax`.

Parameters:
- `data`: Stationary input data matrix of size T x n.
- `kmax`: Maximum number of factors.
- `nbck`: Number of sub-blocks to use. The default is floor(n/10).
- `cmax`: Maximum value for the penalty constant. The default is 3.
- `seed`: Seed for the random permutation generator. The default is 1776.
- `standardize`: Whether to standardize the input data matrix. The default is True.
- `ax`: plt.Axes canvas on which to plot the graphs as in the paper.

Returns:
- `rhat1`, `rhat2`: The estimated number of common factors at the 5% and 1% thresholds, along with the `plt.Axes` if `ax` is passed.

```python
import matplotlib.pyplot as plt
from src.modules.ABC_crit import ABC_crit

## load the data
T, n = data.shape
## Bai and Ng (2002) upper bound for kmax
kf = int((min(n,T)/100)**0.25)
kmax = 8 * kf if kf > 0 else 8

fig, ax = plt.subplots()
rhat1, rhat2, ax = ABC_crit(data, kmax=kmax, ax=ax, standardize=True)
plt.show()
```

### `DFM_periodogram`
This function provides the means for a simple spectral analysis of univariate series.
- The input data has to be a pandas Series object and it cannot contain missing values or else the function throws out an error.
- The frequency of the input data is an optional argument. This argument takes a two character string input ('QD' or 'MD') to determine the frequency of the input data.
- By default the scaling of the output of the function is such that it corresponds to the power spectrum in units of V^2. Can be set to 'density' if the other scaling is desired.
- The default kernel window is the Bartlett kernel. Can be set to any of the valid kernels recognized by scipy.
- The function returns a data class that contains the array of sample frequencies, the estimated periodogram, and the plot of the periodogram against the sample frequencies.

```python
import matplotlib.pyplot as plt
from src.modules.DFM_periodogram import DFM_periodogram

## load the data
univariate_series = data.iloc[:,0] # the first series for instance

fig, ax = plt.subplots()
dfm_p = DFM_periodogram(univariate_series, freq='QD', ax=ax)
plt.show()

## Power spectral density with a Hann window
dfm_p2 = DFM_periodogram(univariate_series, freq='QD', window='hann', scaling='density')
```

### `impute_missing`
This function fills the missing values of a univariate series via a local level model (random walk plus noise) estimated by maximum likelihood. The missing values are replaced with the Kalman smoothed state means, while the observed values are left unchanged.
- The series may contain missing values anywhere, in the interior or at the edges, but it must have at least 2 non-missing observations.

```python
from src.utils.impute import impute_missing

imputed_series = impute_missing(series)
```

### Unit root diagnostics
The module `src/diagnostics.py` contains wrappers and tests that help to decide whether a series is stationary after the FRED transformations. None of the functions accepts series with missing values.
- `adf_wrapper` and `kpss_wrapper` are wrappers around the `adfuller` and `kpss` functions of `statsmodels`. They return the test statistic, the p-value, the number of lags, the number of observations and the critical values as a labeled pandas Series.
- `CT_unitroot` implements the wild bootstrap ADF test of Cavaliere and Taylor (2008) with Rademacher weights, which is robust to nonstationary volatility such as variance breaks and heteroskedasticity.

Parameters of `CT_unitroot`:
- `y`: The level series to test.
- `regression`: Deterministic specification: `'n'`, `'c'`, `'ct'` or `'ctt'`. The default is `'ct'`.
- `B`: Number of bootstrap replications. The default is 999.
- `rng`: `np.random.Generator` instance. This argument is required and must be passed from the main script.
- `**kwargs`: Additional arguments passed to `adfuller` (e.g. `maxlag`, `autolag`).

Returns:
- `dict()` containing:
    - the ADF test statistic
    - bootstrap p-value
    - the lag order selected via BIC
    - the bootstrap scheme
    - the deterministic specification

```python
import numpy as np
from src.diagnostics import adf_wrapper, kpss_wrapper, CT_unitroot

adf_results = adf_wrapper(fred_df['USSERV'], regression='c')
kpss_results = kpss_wrapper(fred_df['USSERV'], regression='c')

rng = np.random.default_rng(seed=1776)
ct_results = CT_unitroot(fred_df['USSERV'], regression='c', B=999, rng=rng)
print(ct_results)
```

## Requirements
- [`numpy>=1.26.0`](https://numpy.org/)
- [`pandas>=2.2.0`](https://pandas.pydata.org/)
- [`scipy>=1.11.0`](https://scipy.org/)
- [`matplotlib>=3.6.0`](https://matplotlib.org/)
- [`seaborn>=0.13.0`](https://seaborn.pydata.org/)
- [`statsmodels>=0.14.0`](https://www.statsmodels.org/)

## References
- Alessi, L., Barigozzi, M., & Capasso, M. (2010). Improved Penalization for Determining the Number of Factors in Approximate Factor Models. *Statistics & Probability Letters*, 80(23–24), 1806–1813.
- Bai, J., & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models. *Econometrica*, 70(1), 191–221.
- Bańbura, M., & Modugno, M. (2014). Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data. *Journal of Applied Econometrics*, 29(1), 133–160.
- Cavaliere, G., & Taylor, A. M. R. (2008). Bootstrap Unit Root Tests for Time Series with Nonstationary Volatility. *Econometric Theory*, 24(1), 43–71.
- McCracken, M. W., & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research. *Journal of Business & Economic Statistics*, 34(4), 574–589.
- McCracken, M. W., & Ng, S. (2021). FRED-QD: A Quarterly Database for Macroeconomic Research. *Federal Reserve Bank of St. Louis Review*, 103(1), 1–44.
- Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python. *Proceedings of the 9th Python in Science Conference*.

## License
This product is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
