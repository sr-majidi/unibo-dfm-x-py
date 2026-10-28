import scipy
from scipy.stats import chi2
import numpy as np
import pandas as pd
import sys
sys.path.insert(-1, './src/')

import matplotlib.pyplot as plt
import seaborn as sns
from src.modules.ABC_crit import ABC_crit
from src.modules.Fred_transform import Fred_transform
from src.utils.impute import impute_missing
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

plt.rc('figure', figsize=(16,6))
sns.set_style('darkgrid')

## Define default seed
seed = 1776

"""
1. Load and initialize the data.
    Data source is FredQD Quarterly data base.

"""

## Load data and apply fred transforms
df = pd.read_csv('./data/2026-04-QD.csv')
## Often for some variables the observation corresponding to the last observation date is not yet recorded
df = df.iloc[:-1,:]
fred_df = Fred_transform(df, freq='QD')
fred_df.index.freq = fred_df.index.inferred_freq

## The first two observations will be lost due to the stationary I(2) transforms
fred_df = fred_df.iloc[2:,:]

## Choose how to handle missing data
ls_missing = fred_df.isna().any()
ls_missing = ls_missing.loc[ls_missing == True]
ls_missing.shape

series_init = dict()

## To get a sense of the starting date for each series
for key, _ in ls_missing.items():
    fr_index = fred_df[key].first_valid_index()
    series_init[key] = fr_index

fred_df.loc[fred_df.isna().any(axis=1), fred_df.isna().any(axis=0)]

## Starting dates vary from one variable to another
## To eliminate the NaNs from the data set, we can move the starting date to a future point in time
## and then remove the remaining columns for which the data set still contains missing values.
## Setting the starting point to 1971Q3 and then removing the remaining columns with NaNs seems to be a good compromise.

fred_df = fred_df.loc['1971-06-01':,:]
fred_df = fred_df.dropna(axis=1)

## Some series are still non-stationary even after transformations
## Need to identify and analyze
nonstat_names = []
for c in fred_df.columns:
    ## Tests are run on series up to 2019 to exclude COVID19 shocks
    adf_pval = adfuller(fred_df[c].loc[:'2019'], autolag='AIC', maxlag=8,regression='c')[1]
    if adf_pval > 0.05:
        nonstat_names.append(c)
        
'''
['USEHS',
 'USSERV',
 'CES9092000001',
 'CIVPART',
 'GS1TB3Mx',
 'NWPIx',
 'GFDEBTNx',
 'TLBSNNBBDIx']

These are the series that do not pass the Dickey Fuller test with constant regression.
Each series has to be individually analyzed via other tests such as kpss, acf, Taylor-Cavaliere, ecc.
'''

## 1. Non stationarity of USEHS is obviously due to structural break in the late 90s.
## 2. USSERV seems to have a unit root.
## 3. CES9092000001 seems to have a unit root.
## 4. CIVPART is not a unit root process but it isn't stationary either as it does not pass the kpss test under any circumstances.
## 5. GS1TB3Mx is evidently not stationary, it is enough too look at its plot over time.
## 6. NWPIx is trend stationary.
## 7. GFDEBTNx is super non stationary. It is obvious that its variance doesn't exist.
## 8. TLBSNNBBDIx is also non stationary.

## Thus, we will keep USEHS and de-trend NWPIx. The rest of the series will be dropped.
nonstat_names.remove('USEHS')
nonstat_names.remove('NWPIx')
fred_df = fred_df.drop(columns=nonstat_names)

## Detrending NWPIx
sm.tsa.tsatools.detrend(fred_df['NWPIx'], order=1).plot()
sm.tsa.tsatools.detrend(fred_df['NWPIx'], order=2).plot()

## quadratic detrending yields smoother results
fred_df['NWPIx'] = sm.tsa.tsatools.detrend(fred_df['NWPIx'], order=2)

## Conviene stimare anche lo spettro di queste serie per valutare l'importanza dei diversi cicli
fxx1, pxx_spec_NWPIx = scipy.signal.periodogram(fred_df['NWPIx'], fs=4, window='bartlett', scaling='spectrum')
fxx2, pxx_spec_USEHS = scipy.signal.periodogram(fred_df['USEHS'], fs=4, window='bartlett', scaling='spectrum')
fig, ax = plt.subplots(nrows=2)
ax[0].plot(fxx1, pxx_spec_NWPIx, label='NWPIx')
ax[0].set_xlabel('Frequency Cycles - 1.00 = 4 Quarters')
ax[0].set_ylabel('Spectral Power - In units of $x^2$')
ax[0].set_title('Sample Periodogram with Bartlett Kernel')
ax[0].legend()

ax[1].plot(fxx2, pxx_spec_USEHS, label='USEHS')
ax[1].set_xlabel('Frequency Cycles - 1.00 = 4 Quarters')
ax[1].set_ylabel('Spectral Power - In units of $x^2$')
ax[1].legend()
plt.show()

## seems like the low frequency components are very dominant even after detrending, thus
## it's best to remove these series as well

fred_df = fred_df.drop(columns=['NWPIx','USEHS'])

## Need to handle COVID period. Typically 2020Q2 and 2020Q3 are set to NaN and later imputed with the Kalman smoother.
fred_df.loc['2020-06-01':'2020-09-01',:] = None
fred_df_full = fred_df.copy(deep=True)
fred_df = fred_df.iloc[:80]

"""
2. Factor Analysis via Static Methods
    2.1 Estiamte the number of common factors via the ABC criterion.
    2.2 Obtain estimates of the loadings and the factors via PCA and linear projection under the identification scheme:
        (F.T @ F)/T = I_r
    2.3 Idnetify the matrix of loadings by fixing the sign of the first row to be positive
    2.4 Obtain residuals and compute their sample variances
    2.5 AIC + Likelihood Ratio test for determining the order of the VAR process followed by the common factors.

"""

T, n = fred_df.shape
## Bai and Ng (2002) upper bound for kmax
kf = int((min(n,T)/100)**0.25)
kmax = 8 * kf if kf > 0 else 8

fig, ax = plt.subplots()
rhat1, rhat2, plot = ABC_crit(fred_df, kmax, seed=seed, ax=ax, standardize=True)
plt.show()

## rhat1 and rhat2 are the estimated number of common factors at 5 and 1 percent thresholds.
kfactors = rhat1

## To proceed, we first standardize the data
fred_df_means = fred_df.mean()
fred_df_stds = fred_df.std()

fred_df = (fred_df - fred_df_means)/fred_df_stds

## PC Analysis via Singular Value Decomposition of the data matrix
U, s, Vh = scipy.linalg.svd(fred_df, full_matrices=False)
V = Vh.T
S = np.diag(s)

## Visualize the scree plot
p = sns.scatterplot(data=s/np.sqrt(T))
p.set(xlabel='Component Index',
      ylabel='$Singular Value (Scaled by \sqrt{T})$',
      title='Scree Plot')

## Principal components are obtained as the linear transformation of the data via the transpose of the matrix of right singular vectors
X_PC = Vh @ fred_df.transpose()

"""
Under the identification assumption E[ff'] = I, we have:
                                      
        X = F Lambda.T + Xi -> (X.T @ X)/T = Lambda ( F.T @ F )/T Lambda.T + (Xi.T @ Xi)/T
            
    From the SVD of the data matrix we have:
            
        X = U @ S @ V.T -> 
                (X.T @ X)/T = V @ (S^2)/T @ V.T
                            = V_x @ (S^2_x)/T @ V_x.T + V_(-x) @ (S^2_(-x))/T @ V_(-x).T
                            
    Where S_x and V_x are the matrix of singular values and the corresponding matrix of the right singular vectors of the x largest singular values
    
    Trivially, the PCA estiamte of the matrix of loadings is obtained as:
        Lambda_hat = (V_x @ S_x)/sqrt(T)
                                      
"""
V_x = V[:,:kfactors]
S_x = S[:kfactors,:kfactors]

Lambda_hat = V_x @ S_x / np.sqrt(T)

"""
    Note that the matrix of loadings and the factors are identified up to an invertible transformation.
    
    x_t = Lambda @ H @ H.T @ f_t + xi_t  is equiv. to x_t = Lambda @ f_t + xi_t,

    where H @ H.T = I.
    
    Therefore, if:
        
        V_x @ (S^2_x)/T @ V_x.T = Lambda_hat @ Lambda_hat.T,
        
    Also:
        
        V_x @ (S^2_x)/T @ V_x.T = Lambda_hat @ H @ H.T Lambd_hat.T
        
    Which implies:
        
        Lambda_hat = (V_x @ S_x)/sqrt(T)  and also  Lambda_hat @ H = (V_x @ S_x)/sqrt(T) @ H
        
    To identify Lambda_hat, we need to fix the rotation matrix H. A common choice which is also
    intuitive is to set H equal to a diagonal matrix composed of the signs of the entries in the first row of Lambda_hat

        H = np.diag( np.sign(Lambda_hat[0,:]) )
        
    This is equivalent to normalizing the laodings of the first variable to be all positive.
    Also observe that
    
        H @ H.T = H.T @ H = I_r
        
    And therefore, under the identifiaction scheme where the loagings of the first variable are all positive, we obtain a global identification of the factors and the matrix of loadings with the latter given by:
        Lambda_hat = (V_x @ S_x)/sqrt(T) @ H

"""

H = np.diag(np.sign(Lambda_hat[0,:]))
Lambda_hat = Lambda_hat @ H

## The factors are obtained via the linear projection of the data onto the range space of the matrix of loadings
"""
    X = F Lambda.T + Xi
    Objective -> minimize Tr(Xi @ Xi.T) w.r.t F <~~ Minimize the sum of residuals over N and T
    
    The minimization problem follows:
        (X - F Lambda.T) (X - F Lambda.T).T = XX.T - F Lambda.T X.T - X Lambda F.T + F Lambda.T Lambda F.T
    
    And the minimzer is found to be the linear projection:
        F_hat = X Lambda_hat (Lambda_hat.T Lambda_hat)^(-1)
        
    Since Lambda_hat is already given in terms of the singular values and right singular vectors we can deduce:
        F_hat = X V_x S_x/sqrt(T) @ H (H.T @ S_x/sqrt(T) V_x V_x.T S_x/sqrt(T) @ H)^(-1) = X V_x @ S_x/sqrt(T) @ H ( H.T @ S_x^2/T @ H )^(-1)

"""
#F_hat = fred_df @ V_x @ np.linalg.inv( S_x/np.sqrt(T) )
F_hat = fred_df @ V_x @ S_x/np.sqrt(T) @ H @ np.linalg.inv( H.T @ S_x**2/T @ H )

## The idiosyncratic component is naturally estimated via the residuals
common_comp = F_hat@Lambda_hat.T
common_comp.columns = fred_df.columns
resid = fred_df - common_comp

## estimated variance of the residuals, this will be later fed into the EM algorithm in the initialization step
resid_var = resid.std()**2

## Specify a VAR process for the estiamted factors and use AIC to select the lag order of the process
state_var = sm.tsa.VAR(F_hat, freq=F_hat.index.freq)
state_lag_res = state_var.select_order(maxlags=8, trend='n')
lag_aic = state_lag_res.aic

## We now compare the AIC selected process with the next most parsimonious process (whose order is simply given by the AIC selected lag order minus one) via the likelihood ratio test
if lag_aic <= 1:
    state_lag = lag_aic
else:
    fit1 = sm.tsa.VAR(F_hat, freq=F_hat.index.freq).fit(maxlags=lag_aic, ic=None, trend='n')
    fit2 = sm.tsa.VAR(F_hat.iloc[1:,:], freq=F_hat.index.freq).fit(maxlags=lag_aic-1, ic=None, trend='n')
    lr_stat = -2 * (fit2.llf - fit1.llf)
    chi_df = F_hat.shape[1] ** 2

    lr_pval = chi2.sf(lr_stat, chi_df)

    state_lag = lag_aic if lr_pval < 0.05 else lag_aic - 1
    
## To obtain a preliminary estimate of the transition matrices, we fit a VAR(state_lag) process to the common factors
state_var_mod = sm.tsa.VAR(F_hat, freq=F_hat.index.freq)
state_var_res = state_var_mod.fit(maxlags=state_lag, ic=None, trend='n')

"""
3. State Space Representation and Estimation of the DFM

    3.1 Define and initialize the model instance
    3.2 Fix a window size for the rolling-window estimation and forecasting scheme
    3.3 Initialize the estimation procedure with the PCA estimates of the model parameters
    3.4 At each iteration produce 1 step ahead and 4 step ahead forecasts of the select individual series and obtain forecast losses
    3.5 

"""

dfm_ssm_model = sm.tsa.DynamicFactorMQ(fred_df, factor_multiplicities=kfactors,
                                   factor_orders=state_lag, standardize=False,
                                   idiosyncratic_ar1=False)
init_params = np.array([])

"""
Even though statsmodels by default uses PCA to initialize the EM algorithm, I still
prefer to feed the initial parameters myself.

    Note that the parameters should be passed in the following order:
        
        1. Matrix of loadings
        2. Transition matrices of the VAR process
        3. Variances and covarinaces of the factors
           *Note that under the identification scheme employed here this is just the lower triangular entries of the identity matrix
        4. Coefficients of the AR process for the idiosyncratic components - if idiosyncratic_ar1 is set to True
        5. Variances of the idiosyncratic components
        
    Finally, note that all matrices should be vectorized and be passed as a flattened array.

"""

## statsmodels orders the loadings variable by variable (loading.0.1->x1, loading.0.2->x1, ...), hence the row-major ravel of Lambda_hat
#init_params = np.append(init_params, Lambda_hat.T.ravel())
init_params = np.append(init_params, Lambda_hat.ravel())
init_params = np.append(init_params, state_var_res.params.to_numpy().T.ravel())
init_params = np.append(init_params, np.identity(kfactors)[np.tril_indices(kfactors,0)])
init_params = np.append(init_params, resid_var.to_numpy())

## The window size is set to 80 quareters, equivalent to 20 years which is standard practice in the literature
W = 80



#dfm_ssm_res = dfm_ssm_model.fit(start_params=init_params, transformed=True,
#                                cov_type='none', method='em')



"""
4. Post-Estimation Analysis and Diagnostic Tests
"""

"""
5. Forecasting via The Common Component
"""




    
    
    
    