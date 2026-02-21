# Behavioral Portfolio Optimizer: Cumulative Prospect Theory 

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Traditional Mean-Variance Optimization (MVO) assumes investors are perfectly rational and view upside volatility and downside volatility equally. However, behavioral economics proves otherwise. 

This repository implements a portfolio optimization engine based on **Cumulative Prospect Theory (CPT)** (Kahneman & Tversky, 1992). It optimizes asset allocation by maximizing "Prospect Value" rather than the Sharpe Ratio, accounting for human loss aversion and the tendency to overweight extreme tail risks.

##  Key Features
* **Loss Aversion Asymmetry:** Implements a penalty for downside drawdowns $\lambda = 2.25$ times greater than equivalent upside gains.
* **Probability Distortion:** Applies CPT probability weighting to account for the human cognitive bias of overestimating the likelihood of rare "fat tail" events.
* **Mean-Variance Benchmarking:** Built-in standard Sharpe maximization to directly compare rational vs. behavioral capital allocations.

##  Mathematical Formulation

### 1. The Value Function
Evaluates gains and losses relative to a reference point (e.g., risk-free rate), heavily penalizing losses:
$$v(x) = \begin{cases} x^\alpha & \text{if } x \ge 0 \\ -\lambda(-x)^\beta & \text{if } x < 0 \end{cases}$$

### 2. Probability Weighting Function
Transforms empirical probabilities to model human cognitive distortion:
$$w(p) = \frac{p^\gamma}{(p^\gamma + (1-p)^\gamma)^{1/\gamma}}$$

## Disclaimer

For Educational and Research Purposes Only.
The code and models in this repository are provided for academic and demonstrative purposes only. They do not constitute financial advice, investment recommendations, or an offer to buy/sell securities. Algorithmic trading and quantitative modeling carry significant risk. The author is not responsible for any financial losses incurred from the use of this software.