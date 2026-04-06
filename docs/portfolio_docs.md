# Risk Management: Monte Carlo Portfolio Simulation

- [1. Overview](#1-overview)
- [2. What the Application Does](#2-what-the-application-does)
- [3. Simulation Methodology](#3-simulation-methodology)
- [4. Risk Metrics](#4-risk-metrics)
- [5. Notes and Limitations](#5-notes-and-limitations)

## 1. Overview

This project is a **Streamlit-based Monte Carlo risk analysis tool** for an equity portfolio. It loads a holdings file, downloads historical adjusted close prices with `yfinance`, estimates the joint dependence structure of asset returns from historical data, and simulates many possible future portfolio paths.

The purpose of the application is to approximate the future distribution of portfolio value,


$$
V_{t+h} = \sum_{i=1}^{N} q_i S_{i,t+h}
$$


where:

- $N$ is the number of assets,
- $q_i$ is the number of shares held in asset $i$,
- $S_{i,t+h}$ is the simulated future price of asset $i$ after horizon $h$.

From the simulated distribution of $V_{t+h}$, the application summarizes downside risk and performance using:

- **Value at Risk (VaR)**
- **Conditional Value at Risk (CVaR)**
- **Sharpe Ratio**
- **Sortino Ratio**

The current implementation combines a Streamlit front end with a simulation engine that computes historical log returns, estimates a correlation matrix, generates correlated shocks, constructs price paths, and visualizes simulated portfolio outcomes.

## 2. What the Application Does

Based on the current codebase, the application performs the following tasks:

### 2.1 Load Portfolio Holdings

The application reads a portfolio CSV file and extracts the holdings needed for simulation. The expected columns include:

- `Ticker`
- `Shares`
- `Market Value [USD]`

If duplicate rows exist for the same ticker, share counts are aggregated before the portfolio paths are constructed.

Mathematically, if a ticker appears multiple times, the total holding is:


$$
q_i = \sum_{j \in \mathcal{J}_i} q_{ij}
$$


where $\mathcal{J}_i$ denotes all rows associated with ticker $i$.

### 2.2 Download Historical Prices

For each ticker in the portfolio, the code downloads adjusted close prices over a user-selected historical window. Let $P_{i,t}$ denote the adjusted close price of asset $i$ at time $t$.

These price histories are the foundation for estimating drift, volatility, and cross-asset dependence.

### 2.3 Compute Historical Log Returns

The code converts prices into daily log returns:


$$
r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
$$


Using log returns is convenient because they aggregate additively over time and map naturally into exponential price dynamics.

From the historical sample, the code estimates:


$$
\mu_i = \frac{1}{T}\sum_{t=1}^{T} r_{i,t}
$$


$$
\sigma_i = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(r_{i,t}-\mu_i)^2}
$$


and the empirical correlation matrix


$$
\mathbf{R} = (\rho_{ij})_{i,j=1}^N
$$


with


$$
\rho_{ij} = \frac{\operatorname{Cov}(r_i,r_j)}{\sigma_i \sigma_j}
$$


### 2.4 Simulate Correlated Future Returns

A key feature of the application is that it does **not** simulate each asset independently. Instead, it preserves the historical cross-sectional dependence structure by generating correlated shocks from the estimated correlation matrix.

Suppose


$$
\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$


where $\mathbf{I}$ is the identity matrix. If the correlation matrix $\mathbf{R}$ is positive definite, then there exists a lower triangular matrix $\mathbf{L}$ such that


$$
\mathbf{R} = \mathbf{L}\mathbf{L}^{\top}
$$


This is the **Cholesky decomposition**.

The code uses this factorization to transform independent standard normal shocks into correlated shocks:


$$
\mathbf{X} = \mathbf{L}\mathbf{Z}
$$


Then

$$
\operatorname{Cov}(\mathbf{X})
= \operatorname{Cov}(\mathbf{L}\mathbf{Z})
= \mathbf{L}\operatorname{Cov}(\mathbf{Z})\mathbf{L}^{\top}
= \mathbf{L}\mathbf{I}\mathbf{L}^{\top}
= \mathbf{L}\mathbf{L}^{\top}
= \mathbf{R}
$$

So this transformation ensures that the simulated shocks inherit the historical correlation structure.

Intuitively:

- $\mathbf{Z}$ starts as independent noise,
- $\mathbf{L}$ mixes the components together,
- the resulting vector $\mathbf{X}$ has the desired dependence pattern.

This matters because portfolio risk is driven not only by each asset’s standalone volatility, but also by how assets move together. If two stocks tend to fall at the same time, independent simulation would understate downside risk.

The implementation also includes a **numerically stable Cholesky routine**. If the estimated correlation matrix is nearly singular or not strictly positive definite due to finite-sample noise, the code adds a small diagonal adjustment:

$$
\mathbf{R}_{\varepsilon} = \mathbf{R} + \varepsilon \mathbf{I}
$$

If this is still insufficient, the matrix is projected to a positive definite approximation using an eigenvalue correction:

$$
\mathbf{R} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\top}
\quad \Longrightarrow \quad
\tilde{\mathbf{R}} = \mathbf{Q}\tilde{\mathbf{\Lambda}}\mathbf{Q}^{\top}
$$

where the adjusted eigenvalue matrix $\tilde{\mathbf{\Lambda}}$ clips very small or negative eigenvalues to a small positive constant. This guarantees that the Cholesky factor exists.

### 2.5 Distribution Choices

After correlated shocks are generated, the application maps them into one of three return models.

#### Normal Distribution

Under the normal specification, each asset return is simulated as

$$
\tilde r_{i,t} \sim \mathcal{N}(\mu_i, \sigma_i^2)
$$

This provides a simple baseline and is easy to interpret, but it may understate extreme tail events.

#### t-Distribution

Under the Student’s t specification, the simulation uses heavier tails. In the current code, the degrees of freedom are fixed at

$$
\nu = 5
$$

A t-distributed shock is scaled so that its dispersion is aligned with the historical volatility estimate. This makes the simulation more conservative in the tails than the normal model.

#### Empirical Distribution

Under the empirical specification, the simulation avoids imposing a parametric distribution. Instead, it uses the probability integral transform idea:

1. generate correlated Gaussian shocks,
2. convert them to uniforms via the normal CDF,
3. map those uniforms into historical return quantiles.

Formally, if

$$
U_{i,t} = \Phi(X_{i,t})
$$

then the empirical return is obtained through the historical quantile function:

$$
\tilde r_{i,t} = F_i^{-1}(U_{i,t})
$$

where $F_i^{-1}$ is the empirical quantile function of asset $i$'s historical returns.

This approach allows the simulated marginal return distribution to reflect observed skewness, fat tails, and other non-Gaussian features present in the sample.

### 2.6 Construct Simulated Price Paths

Once simulated future log returns are obtained, the code converts them into price paths by exponential compounding:

$$
S_{i,t+1} = S_{i,t} \exp(\tilde r_{i,t+1})
$$

Over multiple steps, this becomes

$$
S_{i,t+h} = S_{i,t} \exp\left(\sum_{k=1}^{h} \tilde r_{i,t+k}\right)
$$

This is consistent with standard geometric return dynamics.

### 2.7 Construct Simulated Portfolio Value Paths

At each simulated time step, the portfolio value is computed by summing across all holdings:

$$
V_{t+h}^{(m)} = \sum_{i=1}^{N} q_i S_{i,t+h}^{(m)}
$$

where $m = 1,2,\dots,M$ indexes the Monte Carlo path.

The output is therefore a simulated distribution

$$
\{V_{t+h}^{(1)}, V_{t+h}^{(2)}, \dots, V_{t+h}^{(M)}\}
$$

from which risk statistics can be computed.

### 2.8 Visualize Simulation Results

The application plots:

- all simulated portfolio paths,
- the mean simulated path,
- a 5% to 95% confidence band.

This gives the user a visual representation of both the central tendency and the dispersion of simulated future portfolio values.

### 2.9 Handle Missing Tickers

If price data cannot be downloaded for some holdings, those tickers are excluded from the simulation and reported to the user. This keeps the application robust when external market data is incomplete.

## 3. Simulation Methodology

This section summarizes the full workflow mathematically.

### 3.1 Historical Estimation

Given historical adjusted close prices $P_{i,t}$, compute log returns:

$$
r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
$$

Then estimate:

- mean return vector $\boldsymbol{\mu}$,
- volatility vector $\boldsymbol{\sigma}$,
- correlation matrix $\mathbf{R}$.

### 3.2 Correlated Shock Generation

Let

$$
\mathbf{R} = \mathbf{L}\mathbf{L}^{\top}
$$

be the Cholesky factorization of the empirical correlation matrix. If

$$
\mathbf{Z}_{t}^{(m)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

then correlated shocks are generated by

$$
\mathbf{X}_{t}^{(m)} = \mathbf{L}\mathbf{Z}_{t}^{(m)}
$$

This preserves the empirical dependence structure across assets.

### 3.3 Distribution Mapping

The correlated shocks are converted into uniforms:

$$
\mathbf{U}_{t}^{(m)} = \Phi\bigl(\mathbf{X}_{t}^{(m)}\bigr)
$$

These uniforms are then mapped into the chosen marginal distribution:

- **Normal:**
$$
\tilde r_{i,t}^{(m)} = \Phi^{-1}(U_{i,t}^{(m)}; \mu_i, \sigma_i)
$$

- **t-distribution:**
$$
\tilde r_{i,t}^{(m)} = \mu_i + \sigma_i \cdot g_{\nu}(U_{i,t}^{(m)})
$$
for a suitably scaled t-quantile transform $g_{\nu}(\cdot)$

- **Empirical:**
$$
\tilde r_{i,t}^{(m)} = F_i^{-1}(U_{i,t}^{(m)})
$$

### 3.4 Price Evolution

For each asset and each simulation path,

$$
S_{i,t+1}^{(m)} = S_{i,t}^{(m)} \exp\left(\tilde r_{i,t+1}^{(m)}\right)
$$

### 3.5 Portfolio Aggregation

At each future horizon,

$$
V_{t+h}^{(m)} = \sum_{i=1}^{N} q_i S_{i,t+h}^{(m)}
$$

These values produce the empirical Monte Carlo distribution of future portfolio value.

## 4. Risk Metrics

The current code computes the following metrics from the simulated terminal portfolio values.

### 4.1 Portfolio Return and Loss

Let $V_0$ denote the initial portfolio value and $V_T^{(m)}$ denote the terminal value under simulation path $m$. Then the simulated portfolio return is

$$
R^{(m)} = \frac{V_T^{(m)} - V_0}{V_0}
$$

and the dollar loss is

$$
L^{(m)} = V_0 - V_T^{(m)}
$$

### 4.2 Value at Risk (VaR)

At confidence level $\alpha$, VaR is the empirical $\alpha$-quantile of the loss distribution:

$$
\operatorname{VaR}_{\alpha} = \inf\{\ell \in \mathbb{R} : \mathbb{P}(L \le \ell) \ge \alpha\}
$$

In the current application, the reported confidence level is

$$
\alpha = 0.95
$$

The code reports both:

- dollar VaR,
- percentage VaR, where
$$
\operatorname{VaR}_{\alpha}^{\%} = \frac{\operatorname{VaR}_{\alpha}}{V_0}
$$

### 4.3 Conditional Value at Risk (CVaR)

CVaR measures the **average loss in the tail beyond the VaR cutoff**:

$$
\operatorname{CVaR}_{\alpha} = \mathbb{E}[L \mid L \ge \operatorname{VaR}_{\alpha}]
$$

This is often more informative than VaR because it measures not just the threshold of bad outcomes, but also how severe those bad outcomes are on average.

### 4.4 Sharpe Ratio

The Sharpe Ratio compares average excess return to total volatility:

$$
\operatorname{Sharpe} = \frac{\mathbb{E}[R - R_f]}{\sigma(R - R_f)}
$$

In the current code, the risk-free input defaults to

$$
R_f = 0.03
$$

### 4.5 Sortino Ratio

The Sortino Ratio modifies the Sharpe Ratio by replacing total volatility with downside deviation:

$$
\operatorname{Sortino} = \frac{\mathbb{E}[R - R_f]}{\sigma_{-}(R - R_f)}
$$

where

$$
\sigma_{-}(R - R_f) = \sqrt{\mathbb{E}\left[\min(R-R_f,0)^2\right]}
$$

This metric is especially useful when the user is primarily concerned with downside risk rather than upside fluctuation.

## 5. Notes and Limitations

This document reflects the **current code implementation**. A few important limitations should be noted:

- The application currently focuses on **portfolio-level Monte Carlo simulation**, not trading or execution.
- The code does **not** currently implement buy, sell, or trade-log functionality.
- The code does **not** currently implement **Incremental VaR (IVaR)**.
- The t-distribution degrees of freedom are fixed in code rather than user-configurable.
- The simulation depends heavily on the selected historical window and on the quality of Yahoo Finance data.
- The empirical approach can only reproduce features that already appear in the historical sample.
- Correlation-based dependence modeling is practical and interpretable, but it still simplifies real market dependence, especially in crisis regimes.

---

## Summary

This application is a **Streamlit Monte Carlo portfolio risk tool** that estimates historical return dynamics, preserves cross-asset dependence through a Cholesky-based correlation structure, simulates future portfolio paths under multiple distributional assumptions, and reports VaR, CVaR, Sharpe Ratio, and Sortino Ratio.

Compared with the earlier markdown draft, this revised document matches the current code more closely, removes sections that are unnecessary for the Streamlit workflow, expands the explanation of Cholesky decomposition, and presents the methodology in a more formal mathematical style.
