# Sector Factor Analysis Documentation

- [1. Overview](#1-overview)
- [2. Data Inputs](#2-data-inputs)
- [3. Application Workflow](#3-application-workflow)
- [4. Regression Model](#4-regression-model)
- [5. Output Structure](#5-output-structure)
- [6. Visualization](#6-visualization)
- [7. Streamlit Interface Logic](#7-streamlit-interface-logic)
- [8. Practical Interpretation](#8-practical-interpretation)
- [9. Limitations](#9-limitations)
- [10. Summary](#10-summary)

## 1. Overview

This Streamlit application performs **sector-level factor exposure analysis** for portfolio holdings.  
Given a holdings file with stock tickers and sector classifications, the app allows the user to select a sector and estimate the factor sensitivities of all stocks within that sector.

The workflow combines:

- portfolio holdings data,
- historical factor ETF returns,
- historical market returns,
- stock-level return regressions.

For each stock in the selected sector, the application fits a linear regression model and estimates exposures to:

- **Market factor**: `CSUS.L`
- **Style factors**:
  - `SIZE`
  - `VLUE`
  - `MTUM`
  - `QUAL`

The outputs include:

- a table of estimated factor exposures,
- bar charts for each estimated coefficient.

---

## 2. Data Inputs

The code uses the following fixed inputs:

$$
\text{Factor tickers} = [\text{SIZE}, \text{VLUE}, \text{MTUM}, \text{QUAL}]
$$

$$
\text{Market ticker} = \text{CSUS.L}
$$

$$
\text{Return history} = 1\text{ year}
$$

$$
\text{Return interval} = 1\text{ day}
$$


### 2.1 Portfolio Holdings File

The portfolio file is loaded using:

```python
load_portfolio_data()
```

This file is expected to contain at least the following columns:

- `Ticker`
- `sector`

The `Ticker` column identifies the stocks in the portfolio, and the `sector` column is used to group the holdings into sector-level subsets.

### 2.2 Factor and Market Price Data

The application downloads historical adjusted close data from Yahoo Finance using `yfinance`.

The factor tickers are:

- `SIZE`
- `VLUE`
- `MTUM`
- `QUAL`

The market ticker is:

- `CSUS.L`

The historical prices are converted into daily percentage returns.

For a generic asset price series $P_t$, the daily return is computed as

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}.
$$

In the code, this is implemented using:

```python
pct_change().fillna(0)
```

so that the first missing return observation is replaced by zero.

---

## 3. Application Workflow

The application proceeds through the following steps.

### 3.1 Load Portfolio Holdings

The holdings file is read into a pandas DataFrame:

$$
\mathbf{H} = \{(\text{Ticker}_i, \text{sector}_i)\}_{i=1}^N.
$$

The available sector choices are extracted from the `sector` column and shown in a Streamlit select box.

### 3.2 Download Factor and Market Returns

The function

```python
load_factor_and_market_returns()
```

downloads one year of daily adjusted close prices for the factor ETFs and the market benchmark. These prices are transformed into return series:

$$
r_{f,t}^{(j)} = \frac{P_{f,t}^{(j)} - P_{f,t-1}^{(j)}}{P_{f,t-1}^{(j)}},
\qquad j = 1,2,3,4
$$

for the four style factors, and

$$
r_{m,t} = \frac{P_{m,t} - P_{m,t-1}}{P_{m,t-1}}
$$

for the market factor.

The resulting objects are:

- `factor_returns`
- `market_returns`

### 3.3 Filter Holdings by Sector

Once a sector is selected, the code extracts all tickers in that sector:

$$
\mathcal{S} = \{ i : \text{sector}_i = \text{selected sector} \}.
$$

The corresponding ticker list is then used to download stock price data from Yahoo Finance.

If no tickers are found for the selected sector, the function returns an empty DataFrame.

### 3.4 Download Stock Prices and Compute Stock Returns

For all stocks in the selected sector, the app downloads adjusted close prices and computes daily stock returns:

$$
r_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}.
$$

Missing prices are forward-filled before returns are computed:

$$
P_{i,t}^{\text{filled}} = \text{ffill}(P_{i,t}).
$$

After computing returns, the code removes invalid columns:

- stocks with all missing returns,
- stocks with zero absolute return sum.

This ensures the regression is only run on stocks with usable historical data.

### 3.5 Merge All Return Series

The code concatenates:

- factor returns,
- market returns,
- stock returns

into a single aligned DataFrame:

$$
\mathbf{R}_t =
\left[
r_{m,t},
r_{SIZE,t},
r_{VLUE,t},
r_{MTUM,t},
r_{QUAL,t},
r_{1,t},
\dots,
r_{K,t}
\right].
$$

Rows with missing factor data are removed:

$$
\mathbf{R}_t \leftarrow \mathbf{R}_t \text{ with valid factor observations only}.
$$

Any remaining missing values are filled with zero.

---

## 4. Regression Model

For each stock in the selected sector, the application fits a linear regression model of the form

$$
r_{i,t}
=
\alpha_i
+
\beta_i r_{m,t}
+
\gamma_{i,\text{SIZE}} r_{\text{SIZE},t}
+
\gamma_{i,\text{VLUE}} r_{\text{VLUE},t}
+
\gamma_{i,\text{MTUM}} r_{\text{MTUM},t}
+
\gamma_{i,\text{QUAL}} r_{\text{QUAL},t}
+
\varepsilon_{i,t}.
$$

Here:

- $r_{i,t}$ is the return of stock $i$ on day $t$,
- $\alpha_i$ is the intercept,
- $\beta_i$ is the loading on the market return,
- $\gamma_{i,\cdot}$ are the loadings on the style factors,
- $\varepsilon_{i,t}$ is the regression residual.

### 4.1 Matrix Form

Let the design matrix be

$$
\mathbf{X} =
\begin{bmatrix}
1 & r_{m,1} & r_{\text{SIZE},1} & r_{\text{VLUE},1} & r_{\text{MTUM},1} & r_{\text{QUAL},1} \\
1 & r_{m,2} & r_{\text{SIZE},2} & r_{\text{VLUE},2} & r_{\text{MTUM},2} & r_{\text{QUAL},2} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
1 & r_{m,T} & r_{\text{SIZE},T} & r_{\text{VLUE},T} & r_{\text{MTUM},T} & r_{\text{QUAL},T}
\end{bmatrix}.
$$

For stock $i$, define the response vector

$$
\mathbf{y}_i =
\begin{bmatrix}
r_{i,1} \\
r_{i,2} \\
\vdots \\
r_{i,T}
\end{bmatrix}.
$$

Then the regression can be written as

$$
\mathbf{y}_i = \mathbf{X}\boldsymbol{\theta}_i + \boldsymbol{\varepsilon}_i,
$$

where

$$
\boldsymbol{\theta}_i =
\begin{bmatrix}
\alpha_i \\
\beta_i \\
\gamma_{i,\text{SIZE}} \\
\gamma_{i,\text{VLUE}} \\
\gamma_{i,\text{MTUM}} \\
\gamma_{i,\text{QUAL}}
\end{bmatrix}.
$$

The model is estimated using `LinearRegression(fit_intercept=True)` from scikit-learn.

### 4.2 Interpretation of Coefficients

For each stock:

- **Alpha** measures the average return unexplained by the included factors.
- **Beta** measures sensitivity to the market benchmark `CSUS.L`.
- **SIZE** measures exposure to the size factor.
- **VLUE** measures exposure to the value factor.
- **MTUM** measures exposure to the momentum factor.
- **QUAL** measures exposure to the quality factor.

A positive coefficient means the stock tends to move in the same direction as that factor.  
A negative coefficient means the stock tends to move opposite to that factor, after controlling for the other regressors.

---

## 5. Output Structure

The regression results are stored in a DataFrame named `exposures`.

Its rows correspond to stock tickers, and its columns are:

- `Alpha`
- `Beta`
- `SIZE`
- `VLUE`
- `MTUM`
- `QUAL`

Formally, the output table can be represented as

$$
\mathbf{E} =
\begin{bmatrix}
\alpha_1 & \beta_1 & \gamma_{1,\text{SIZE}} & \gamma_{1,\text{VLUE}} & \gamma_{1,\text{MTUM}} & \gamma_{1,\text{QUAL}} \\
\alpha_2 & \beta_2 & \gamma_{2,\text{SIZE}} & \gamma_{2,\text{VLUE}} & \gamma_{2,\text{MTUM}} & \gamma_{2,\text{QUAL}} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
\alpha_K & \beta_K & \gamma_{K,\text{SIZE}} & \gamma_{K,\text{VLUE}} & \gamma_{K,\text{MTUM}} & \gamma_{K,\text{QUAL}}
\end{bmatrix}.
$$

The app displays this table directly in Streamlit with:

```python
st.dataframe(exposures)
```

---

## 6. Visualization

After the exposures are computed, the application generates a bar chart for each coefficient:

- Alpha
- Beta
- SIZE
- VLUE
- MTUM
- QUAL

For a given factor $f$, the chart plots:

- x-axis: stock ticker
- y-axis: estimated exposure to factor $f$

Thus, for each factor, the visualization shows the cross-sectional dispersion of exposures within the selected sector.

This is useful for identifying:

- which names are most sensitive to the market,
- which names have strong style tilts,
- whether a sector contains heterogeneous factor behavior across its constituents.

---

## 7. Streamlit Interface Logic

The user interface follows a simple sequence:

1. Load the holdings file.
2. Load factor and market return data.
3. Display a sector selection dropdown.
4. Wait for the user to click **Run Factor Analysis**.
5. Compute stock-by-stock regression exposures for the selected sector.
6. Show the resulting table and charts.

If the resulting exposure table is empty, the app displays:

```python
st.warning("No valid exposure results for this sector.")
```

Otherwise it displays a success message and the results.

---

## 8. Practical Interpretation

This application is useful for **cross-sectional factor diagnostics within a sector**.

For example, within the Technology sector, the model can reveal that:

- some stocks have high market beta,
- some names are tilted toward momentum,
- some names behave more like value stocks,
- some names carry relatively high alpha.

This allows the user to compare stocks within the same sector on a more systematic basis.

From a portfolio construction perspective, this can help answer questions such as:

- Which stocks in this sector have the strongest market sensitivity?
- Which names contribute to a momentum tilt?
- Are there stocks with unusually high value or quality exposure?
- Is the sector internally homogeneous, or does it contain distinct factor subgroups?

---

## 9. Limitations

Several limitations should be noted.

### 9.1 Short Estimation Window

The model uses only one year of daily data:

$$
T \approx 252.
$$

This may produce unstable coefficients, especially for volatile stocks.

### 9.2 Linear Specification

The model assumes a linear factor structure:

$$
r_{i,t} = \alpha_i + \mathbf{x}_t^\top \boldsymbol{\beta}_i + \varepsilon_{i,t}.
$$

This may miss nonlinear relationships, regime changes, or interaction effects.

### 9.3 Data Quality Dependence

The results depend on Yahoo Finance data availability and ticker coverage.  
If a stock has incomplete or invalid price history, it may be excluded.

### 9.4 Factor Proxy Choice

The factors are represented by ETF tickers rather than by canonical academic factor portfolios.  
Therefore, the estimated coefficients should be interpreted as **exposure to the chosen tradable proxies**, not necessarily to exact academic factor definitions.

### 9.5 No Statistical Diagnostics Reported

The current code reports coefficient estimates only. It does not display:

- $R^2$,
- t-statistics,
- p-values,
- standard errors,
- confidence intervals.

So the output is best interpreted as an exploratory exposure analysis rather than a formal inference report.

---

## 10. Summary

This Streamlit application estimates **stock-level factor exposures within a user-selected sector** by regressing each stock’s daily return on:

$$
[\text{Market}, \text{SIZE}, \text{VLUE}, \text{MTUM}, \text{QUAL}].
$$

The application:

- loads holdings data,
- downloads historical factor, market, and stock prices,
- computes daily returns,
- fits a linear regression for each stock,
- displays the estimated exposures in both tabular and chart form.

The final result is a practical sector-by-sector factor analysis tool that helps the user understand how individual holdings align with market and style-factor behavior.
