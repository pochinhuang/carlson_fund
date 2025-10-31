# Risk Management: Monte Carlo Analysis

**University of Minnesota Twin Cities**   
**Master of Financial Mathematics**

<small>**Authors:** Ioannis Petropoulos, Po-chin Huang</small>  
<small>**Advisor:** Dr. Thomas Gebhart</small>

---

- [1. Overview](#1.-Overview)
- [2. What is Standard Deviation in a Portfolio?](#2.-What-is-Standard-Deviation-in-a-Portfolio?)
- [3. Monte Carlo Simulation and Distribution Options](#3.-Monte-Carlo-Simulation-and-Distribution-Options)
- [4. What is VaR, CVaR, and IVaR?](#4.-What-is-VaR,-CVaR,-and-IVaR?)
- [5. How to Use the Code?](#5.-How-to-Use-the-Code?)

## 1. Overview

This program helps us understand how risky the portfolio becomes when we add a new stock to it. We give it the file with the current stock positions (from CapIQ), and it does the following:

  - Calculates how much the portfolio’s risk (measured by standard deviation) changes when we add a new stock.
  - Tells us if the fund can afford to buy the stock based on how much cash it has.
  - Simulates thousands of possible future portfolio outcomes using Monte Carlo simulations under different assumptions:
    - Normal distribution (bell curve)
    - t-distribution (fatter tails for extreme cases)
    - Empirical distribution (based on actual past data)
  - Estimates Value at Risk (VaR) and Conditional Value at Risk (CVaR) — which tell us how much we could lose in worst-case scenarios.
  - Calculates Incremental VaR (IVaR) — how much more (or less) risk is added by including the new stock.
  - Also calculates how much each stock in our portfolio contributes to the overall risk, based on how it has behaved in the past vs. the simulated results.

## 2. What is Standard Deviation in a Portfolio?
Standard deviation is a measure of how much returns can go up or down from their average. In finance, it’s used to describe risk or volatility.

When looking at a portfolio (a group of stocks), we calculate:
  - How much the whole portfolio's returns vary using all stocks (with the new one included)
  - How much they vary without the new stock

This helps us answer: Will adding this stock make the portfolio riskier or safer?

The formula looks like this:
σₚ = √(wᵀ Σ w)

Where:
- σₚ is the portfolio standard deviation
- w is a list of how much money is in each stock (weights)
- Σ is a matrix showing how each stock moves with every other stock (their relationships)

We don’t need to solve this by hand — the program does it for us!

## 3. Monte Carlo Simulation and Distribution Options
What is a Monte Carlo Simulation?
A Monte Carlo simulation is a method used to understand uncertainty by generating thousands of possible future outcomes based on random samples. In finance, we use it to estimate how a portfolio might perform under many different market conditions.

Instead of just looking at past data, we simulate many alternate futures. This helps answer questions like:
- What’s the range of possible returns I might get?
- How often might I experience large losses?
- What’s the risk if I add a new stock to my portfolio?

Why Use It?
- Traditional formulas assume perfect conditions (like normal distributions or independence), but real markets are messy.
- Monte Carlo lets you stress test your portfolio under different assumptions.
- It’s especially useful when adding new positions or testing nonlinear risks.

Distribution Options in Monte Carlo:
1. <u>Normal Distribution</u>

    What it assumes:
    - Returns follow a bell-shaped curve (symmetrical, no skew).
    - Most returns are close to the average.
    - Extreme gains or losses are rare.

    When to use:
    - For quick, simple analysis when returns are stable.
    - When you just want a baseline or comparison case.

    Pros:
    - Easy to understand and implement.
    - Works well when historical returns are close to normal.

    Cons:
    - Underestimates tail risk (big losses are too rare).
    - Doesn’t capture real-world events like crashes or booms.

2. <u>t-Distribution</u>

    What it assumes:
    - Similar to normal, but with fatter tails — meaning more chances of extreme outcomes.
    - Accounts for the fact that big jumps or crashes happen more often than a normal curve predicts.

    When to use:
    - When we want to be more cautious about risk.
    - Useful during volatile markets or crisis periods.

    Pros:
    - More realistic than normal for capturing risk.
    - Better for estimating VaR and CVaR, especially in portfolios with high volatility.

    Cons:
    - Slightly more complex to work with.
    - Still assumes returns are symmetric and may not reflect skewness.

3. <u>Empirical Distribution</u>

    What it assumes:
    - No assumptions — it uses actual past return data and randomly samples from it.
    - Simulates what has really happened, without fitting it to a curve.

    When to use:
    - When we want to reflect real market behavior.
    - Great when we have a large, high-quality dataset.

    Pros:
    - Captures real-world patterns like skew, jumps, and volatility clusters.
    - No need to assume a specific shape (e.g., bell curve).

    Cons:
    - Limited to what’s happened before — doesn’t simulate new, unseen events.
    - If the dataset is small or biased, results may be misleading.
    
## 4. What is VaR, CVaR, and IVaR?
These are measures of how much you could lose in bad market conditions.

**Value at Risk (VaR)**:
This tells us: “How much could we lose, with 95% confidence, in a worst-case scenario over a certain time?”

Formula:
$\text{VaR}_\alpha(X) = \inf \left\{ x \in \mathbb{R} \;:\; \mathbb{P}(X + x < 0) \leq \alpha \right\}$

**Conditional Value at Risk (CVaR)**:
CVaR goes one step further: “If things get really bad (past the VaR), what’s our average loss in those extreme cases?”

Formula:
$\text{CVaR}_\alpha(X) = \mathbb{E}\left[ X \mid X \leq \text{VaR}_\alpha(X) \right]$

**Incremental VaR (IVaR)**:
This tells us how much extra risk you’re adding by including a specific stock in the portfolio.

Formula: 
$\text{IVaR} = \text{VaR}_{\text{with}} - \text{VaR}_{\text{without}}$

If the IVaR is positive → adding the stock increases risk.
If it's negative → the stock reduces overall risk.

## 5. How to Use the Code?

### Download the File

1. Log in to Capital IQ and navigate to **Portfolio** → **Portfolio Dashboard**.
![img_01](images/img_01.png)

2. Click on **19-CG001**.
![img_02](images/img_02.png)

3. Select the desired date (make sure to remember the selected date, as you will need it later when entering parameters) and click the EXPORT PORTFOLIO TO CSV icon in the **top right corner**.
![img_03](images/img_03.png)

4. Save the downloaded CSV file into the `csv_files` folder.
![img_04](images/img_04.png)

### Run the Code

Select the cell you want to run and press `Ctrl + Enter`.

1. **Always run the cell below first**, as it contains the required functions:

<div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f8f8f8">

```python
import utils
```
</div>

2. Input the following parameters and run this cell:

    - `file_path`: Enter the path and filename of the CSV file you just downloaded.

    - `distribution`: Choose a distribution type. (For details about distribution options, refer to the earlier section.)

    - `forecast_days`: Specify the number of days you want to forecast.

    - `start_date`: Set this to **one year before your `end_date`**. (Example: If `end_date = "2025-03-29"`, then `start_date = "2024-03-29"`.)

    - `end_date`: Use the date you selected when exporting the file from Capital IQ.


<div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f8f8f8">

```python
# parameters

# file path of CGF holdings from CapIQ
file_path = "csv_files/19-CG001–Holdings.csv"

# distribution choice:
# normal distribution: "n"
# t distribution: "t"
# empirical distribution: "e"
distribution = "e"

# forcast days
forcast_days = 5

# lookback peroid
# 1 year before "end_date"
start_date = "2024-03-29"

# date of CGF holdings file
end_date = "2025-03-29"
```
</div>

3. After running the parameters cell, or whenever you make changes to the portfolio and want to reset, run the following cell:

<div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f8f8f8">

```python
# start and reset
utils.start_reset(file_path)
```
</div>

4. This function allows users to **sell stocks** from the portfolio. If the stock is **not found** in the portfolio, or if the number of shares to be sold **exceeds the current holdings**, a **failure message** will be displayed.

<div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f8f8f8">

```python
# sell position

# ticker of stock to be sold
sell_ticker = "ADUS"
# shares of stock to be sold
sell_shares = 1

utils.sell(sell_ticker, sell_shares)
```
</div>

5. The `utils.ivar(new_ticker, shares, start_date, end_date)` function is used to calculate the **Incremental Value at Risk (IVaR)**.  Similarly, if the number of shares the user intends to purchase exceeds the available cash, the transaction will fail and an **error message** will be displayed.


<div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f8f8f8">

```python
# IVaR

# the ticker of new name that user wants to evaluate
new_ticker = "AAPL"

# shares of the new name
shares = 100

utils.ivar(new_ticker, shares, start_date, end_date)

```
</div>

6. Users can use `utils.print_trade_log()` to **review their trade history**.  If you wish to **reset the portfolio**, simply run `utils.start_reset(file_path)` again.

<div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f8f8f8">

```python
# print trade log
utils.print_trade_log()
```
</div>

6. The `utils.simulation(distribution, forecast_days, start_date, end_date)` function visualizes the simulation results. It also provides the estimated **VaR**, **CVaR**, and identifies the **Top 5 Portfolio Value Volatility Contributors**.

<div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f8f8f8">

```python
utils.simulation(distribution, forcast_days, start_date, end_date)

```
</div>