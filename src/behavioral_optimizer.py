import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class BehavioralOptimizer:
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.0):
        self.returns = returns
        self.rf = risk_free_rate
        self.n_assets = returns.shape[1]
        self.n_obs = returns.shape[0]

        self.alpha = 0.88      
        self.beta = 0.88       
        self.lambda_ = 2.25    
        self.gamma_plus = 0.61 
        self.gamma_minus = 0.69 

    def _value_function(self, x: np.ndarray) -> np.ndarray:
        v = np.zeros_like(x)
        gains = x >= 0
        losses = x < 0
        
        v[gains] = x[gains] ** self.alpha
        v[losses] = -self.lambda_ * ((-x[losses]) ** self.beta)
        return v

    def _probability_weight(self, p: np.ndarray, gamma: float) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            w = (p ** gamma) / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        return np.nan_to_num(w)

    def prospect_objective(self, weights: np.ndarray) -> float:
        port_returns = self.returns.dot(weights).values - self.rf

        sorted_indices = np.argsort(port_returns)
        sorted_returns = port_returns[sorted_indices]

        p_empirical = np.arange(1, self.n_obs + 1) / self.n_obs

        v = self._value_function(sorted_returns)

        decision_weights = np.zeros_like(sorted_returns)

        loss_mask = sorted_returns < 0
        if np.any(loss_mask):
            p_losses = p_empirical[loss_mask]
            w_losses = self._probability_weight(p_losses, self.gamma_minus)
            # Decision weight is the difference in transformed probabilities
            w_losses_diff = np.insert(np.diff(w_losses), 0, w_losses[0])
            decision_weights[loss_mask] = w_losses_diff

        gain_mask = sorted_returns >= 0
        if np.any(gain_mask):
            # Reverse cumulative probabilities for gains
            p_gains = 1 - (np.arange(0, np.sum(gain_mask)) / np.sum(gain_mask))
            w_gains = self._probability_weight(p_gains, self.gamma_plus)
            w_gains_diff = np.insert(-np.diff(w_gains), 0, w_gains[0] - self._probability_weight(p_gains[1], self.gamma_plus) if len(p_gains)>1 else w_gains[0])
            decision_weights[gain_mask] = w_gains_diff

        prospect_value = np.sum(decision_weights * v)

        return -prospect_value

    def sharpe_objective(self, weights: np.ndarray) -> float:
        port_returns = self.returns.dot(weights)
        mean_ret = port_returns.mean() * 252
        volatility = port_returns.std() * np.sqrt(252)
        sharpe = (mean_ret - self.rf) / volatility
        return -sharpe

    def optimize(self, method: str = 'prospect') -> dict:
        init_guess = np.array([1.0 / self.n_assets] * self.n_assets)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        if method == 'prospect':
            obj_func = self.prospect_objective
        else:
            obj_func = self.sharpe_objective

        result = minimize(
            obj_func, 
            init_guess, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'ftol': 1e-6, 'disp': False}
        )
        
        return {
            'weights': np.round(result.x, 4),
            'success': result.success,
            'objective_value': -result.fun
        }


def main():
    print("Fetching historical data...")
    tickers = ['SPY', 'TLT', 'SHV', 'GLD']
    data = yf.download(tickers, start="2018-01-01", end="2024-01-01", progress=False)['Adj Close']

    returns = data.pct_change().dropna()

    returns = returns[tickers]

    print(f"Data fetched. Shape: {returns.shape}")
    print("Running Optimizations...\n")

    optimizer = BehavioralOptimizer(returns, risk_free_rate=0.0)

    mv_result = optimizer.optimize(method='sharpe')
    mv_weights = mv_result['weights']

    pt_result = optimizer.optimize(method='prospect')
    pt_weights = pt_result['weights']

    print("--- Portfolio Allocation Comparison ---")
    print(f"{'Asset':<10} | {'Mean-Variance':<15} | {'Prospect Theory':<15}")
    print("-" * 45)
    for i, ticker in enumerate(tickers):
        print(f"{ticker:<10} | {mv_weights[i]:<15.2%} | {pt_weights[i]:<15.2%}")

    x = np.arange(len(tickers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mv_weights, width, label='Mean-Variance (Max Sharpe)', color='#1f77b4')
    ax.bar(x + width/2, pt_weights, width, label='Prospect Theory (Max Utility)', color='#d62728')

    ax.set_ylabel('Allocation Weight')
    ax.set_title('Portfolio Allocation: Rational (Sharpe) vs Behavioral (Prospect Theory)')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()